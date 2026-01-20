import streamlit as st
import io
import pandas as pd
from PyPDF2 import PdfReader
from openai import OpenAI
import json
import time

# --- Configuration ---
MODEL_NAME = "gpt-4o-mini"
# Note: The OpenAI client handles the base URL automatically.

# --- HARDCODED API KEY (User Request) ---
# IMPORTANT: This key was provided by the user.
OPENAI_API_KEY = "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"

st.set_page_config(page_title="PDF Data Extractor", layout="centered")

st.title("📄 PDF → Structured CSV using LLM API")
st.markdown(
    """
    This tool extracts data from a PDF test sheet and formats the output into a **strict two-column (Label,Value)**
    structure. **The Test Equipment section is the sole exception, outputting four columns** at the very end of the file. 
    All internal commas in values are replaced with semicolons (;) to maintain structure.
    """
)

# --- Upload PDF ---
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

# Initialize filename variable to a default value
default_filename = "extracted_structured_data.csv"
download_filename = default_filename

# Check for uploaded file
if uploaded_file:
    # --- Debugging Status ---
    st.info("Step 1: File uploaded successfully. Proceeding to PDF text extraction.")

    # --- Extract text from PDF ---
    try:
        reader = PdfReader(uploaded_file)
        text_parts = []
        # Limiting to first 5 pages for efficiency and avoiding huge texts
        for i, page in enumerate(reader.pages):
            if i >= 5:
                break
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)

        full_text = "\n".join(text_parts)
        # Limiting the total text sent to the LLM to 12,000 characters
        text_to_send = full_text[:12000]

        if not text_to_send:
            st.error("Could not extract any readable text from the PDF.")
            st.stop()

        st.success("✅ PDF text extracted.")
        st.info("Step 2: Text extraction complete. Preparing API call...")

    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        st.stop()

    # --- FILENAME INPUT ---
    download_filename = st.text_input(
        "Enter desired CSV filename",
        value=default_filename,
        key='csv_filename_input'  # Explicit key ensures the value persists across reruns
    )
    # Ensure it always ends with .csv
    if download_filename and not download_filename.lower().endswith('.csv'):
        download_filename = download_filename + '.csv'
    # -----------------------------------

    # --- Build the prompt and payload ---

    # The system instruction tells the model *how* to behave (its persona and rules)
    system_prompt = """
You are a highly reliable Structured Data Extractor. Your sole output MUST be the requested CSV text. 
Do not include any introductory text, markdown formatting blocks (like ```csv), or explanations. 
***CRITICAL RULE: THE ENTIRE CSV OUTPUT MUST BE COMPACT. DO NOT ADD ANY SPACES BEFORE OR AFTER THE COMMAS THAT SEPARATE COLUMNS. Example: Label,Value (NO SPACES anywhere around the comma).***
***CRITICAL RULE: BEFORE OUTPUTTING ANY ROW, IF THE VALUE FIELD CONTAINS A COMMA (,), YOU MUST REPLACE THAT COMMA WITH A SEMICOLON (;) TO PRESERVE THE COLUMN STRUCTURE. THIS APPLIES TO COMMENTS, NAMEPLATE DATA, AND ALL OTHER VALUE FIELDS.***
***CRITICAL RULE: ALL extracted LABEL and VALUE content MUST be aggressively trimmed of ALL leading and trailing whitespace. The VALUE field must start immediately after the comma and end immediately before the next comma or the end of the line.***
***CRITICAL RULE: YOU MUST EXTRACT EVERY PIECE OF DATA. DO NOT OMIT ANY LABEL, CHECKBOX, NOTE, OR VALUE. EXTRACT EVERYTHING.***

The output CSV MUST adhere strictly to a **TWO-COLUMN** format (Label,Value) across ALL sections except for single-column separator rows AND the final Test Equipment section, which must be four columns.
"""

    # The user query tells the model *what* to do
    user_query = f"""
    Extract ALL tabular and key metadata from the following PDF text. Do not omit any relevant data.
    Your output must be a single, clean CSV string with a strict structure.

    CRITICAL CSV STRUCTURE RULES:

    The output must strictly follow this pattern: Label,Value. If a required value is NOT found, you MUST leave the Value cell BLANK.
    ***CRITICAL: NEVER GUESS OR HALLUCINATE data. If Model or MFR Date are missing, leave the Value cell BLANK.***
    ***Ensure ALL extracted values are clean and free of extra spaces.***

    --- SECTION 1: FIXED METADATA (Rows 1-8, Strict 2 Columns: Label,Value) ---
    - These rows MUST be dedicated to the eight key metadata fields in a Label,Value format (NO trailing comma).
    - **Row 1: Title, [Extracted Title Value - MUST be formatted in Proper Case, e.g., 'MV Vacuum Circuit Breaker']**
    - Row 2: Client,[Extracted Client Value]
    - Row 3: Location,[Extracted Location Value]
    - Row 4: Equipment ID,[Extracted Equipment ID Value]
    - Row 5: Technician(s),[Extracted Technician(s) Value]
    - Row 6: Model,[Extracted Model Value]
    - **Row 7: MFR Date,[Extracted MFR Date Value]**
    - **Row 8: S/N,[Extracted Serial Number Value]**

    - Row 9 MUST contain a header row for the Nameplate data (Strictly 1 Column):
      ---NAMEPLATE DATA---

    - Row 10 ONWARDS: Extract ALL other clearly defined, structured Nameplate Data fields (e.g., Rated Voltage, Interrupting Capacity, Frequency, Closing Volts, etc.) found near the top of the document. Each field must be a separate row in Label,Value format (NO trailing comma).

    - **New Section: ENVIRONMENTAL CONDITIONS**
    - The next row MUST contain the environmental header row (Strictly 1 Column):
      ---ENVIRONMENTAL CONDITIONS---

    - Following the header, extract all three environmental fields in Label,Value format (NO trailing comma):
        Conditions,[Extracted Value e.g., Indoor/Outdoor]
        Ambient Temperature,[Extracted Value]
        Relative Humidity,[Extracted Value]


    --- SECTION 2A: TIMING AND COIL DATA (Conditional Extraction, 2 Columns: Label,Value) ---
    - The next row MUST contain the timing header row (Strictly 1 Column):
      ---TIMING & COIL DATA---

    - **CRITICAL CHANGE: ONLY extract timing and coil data if a test value is present in the source text. Do NOT generate blank rows for tests that were not performed.**
    - You MUST identify the Test (Timing/Coil Current) and the Specific Actuator (Pole 1, Pole 2, Pole 3, Open Coil 1, Close Coil, etc.).
    - If timing/coil data is found, flatten it into the unique Label,Value pairs. Example: Timing - Open Coil 1 - Pole 1,[Value]
    - If NO timing or coil data is found, the ONLY output for this section is the header row above (`---TIMING & COIL DATA---`).

    --- SECTION 2B: MAIN DATA (Catch-All for ALL Remaining Data, Strict 2 Columns: Label,Value) ---
    - The next row MUST contain the headers for the main data section (Strictly 2 Columns):
      Label,Value

    - **ULTIMATE CATCH-ALL MANDATE (FLATTEN EVERYTHING):** Starting immediately after the Timing data, you MUST extract **EVERY SINGLE REMAINING ITEM** of data in the PDF text before the Test Equipment section. If multiple labels and values appear on a single line of the PDF text, you MUST FLATTEN them into separate rows in the Label,Value format. This includes:
        - All inspection notes, checkbox results (e.g., "Cleaned, Acceptable"), counter readings, and test results (like resistance, withstand, etc.).
        - **CRUCIALLY, FOR COMMENTS/NOTES:** Extract all comments, notes, signatures, and date/time stamps. If a comment spans across multiple columns in the source text, **CONCATENATE all data into the single Value cell, replacing any internal commas with a semicolon (;).** The output must **NEVER** have more than two columns (Label,Value). Name the label appropriately (e.g., "General Comment 1", "Signature Line", "Date Tested").

    - If the text clearly indicates the start of a new major test or specification section (e.g., "Insulation Resistance," "Contact Resistance," "PHYSICAL & MECHANICAL INSPECTION"), you MUST insert a separator row first (Strictly 1 Column).

    - **Separator Row Format (Strictly 1 Column):**
      ---[NAME OF TEST SECTION]----

    - Following the separator, all corresponding measurements MUST be presented in the 2-column Label,Value format (NO trailing comma).

    ***CRITICAL MAPPING INSTRUCTION for Multi-Column Data (general rule):***
    - If the source PDF data is a table (e.g., test results) that has multiple values associated with a single label, you MUST FLATTEN this data by appending the column/pole name to the label.
    - Example (NO trailing comma):
        Resistance - Pole 1,[Value 1]
        Resistance - Pole 2,[Value 2]
    - Only use two columns of data.

    --- SECTION 3: TEST EQUIPMENT (LAST SECTION, Strict 4 Columns) ---
    - After all main data is extracted, this is the absolute final section of the output.

    - The next row (Header) MUST contain the 4-column headers (Strictly 4 Columns, NO Trailing Comma):
      Test Equipment,Make/Model,S/N,CAL DUE

    - Following rows (Test Equipment List - Strict 4 Columns, NO Trailing Comma):
      - List the actual equipment data, ensuring the four columns align perfectly.
      - **CRITICAL CORRECTION:** You MUST replace any internal commas in the extracted values (like "Vacuum Interrupters, Inc...") with a semicolon (;) to prevent breaking the four-column structure.
      - No more, no less than 4 columns (Test Equipment,Make/Model,S/N,CAL DUE).

    PDF Text to Analyze:
    {text_to_send}
    """

    # --- Initialize and call OpenAI client ---
    st.info(f"Step 3: Calling OpenAI API ({MODEL_NAME}). This may take a moment...")

    # --- Standard Loading Indicator ---
    with st.status("Analyzing PDF and generating CSV...", expanded=True) as status:
        st.write("Sending request to LLM...")

        try:
            # Initialize client using the hardcoded key
            client = OpenAI(api_key=OPENAI_API_KEY)

            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query}
                ]
            )

            csv_text = response.choices[0].message.content.strip()

            status.update(label="✅ API call successful! Data received.", state="complete", expanded=False)
            st.info("Step 4: Processing received data...")


        except Exception as e:
            # This is the critical block: it catches the failure and prints the error message.
            status.update(label="❌ API Call Failed", state="error", expanded=True)
            st.error(
                f"OpenAI API Call Failed. This usually means the API key is expired, invalid, or you've hit a rate limit/quota issue.")
            st.error(f"Full Error Details: {e}")
            st.stop()  # Halts the script to prevent subsequent code execution

    # --- Convert to DataFrame and download ---
    try:
        # We use header=None because the headers are mixed/custom.
        # Pandas will automatically determine the necessary columns (up to 4 in this case).
        df = pd.read_csv(io.StringIO(csv_text), header=None)

        st.info("Step 5: Data parsed into DataFrame. Displaying preview...")

        st.subheader("Extracted Data Preview (Full CSV Structure)")
        st.dataframe(df)

        # --- DOWNLOAD BUTTON REMAINS HERE ---

        # Convert the full DataFrame back to CSV for download
        # CRITICAL: We explicitly set sep=',' and ensure no extra whitespace is added by pandas here.
        clean_csv = df.to_csv(
            index=False,
            header=False,
            sep=',',
            quoting=1
        ).encode("utf-8")

        # Use the user-specified filename
        st.download_button(
            label="⬇️ Download Structured CSV",
            data=clean_csv,
            file_name=download_filename,
            mime="text/csv"
        )

        # --- Post-Download Confirmation/Thank You (without GIF) ---
        st.markdown("---")
        st.markdown("# **Thanks for using Paper Shredder!**")
        st.markdown(
            f"***Note:*** *The final CSV now includes the two-column structure for all main data, followed by the required four-column Test Equipment section at the end.*")

    except pd.errors.ParserError as e:
        st.warning(
            f"Could not parse LLM output into a DataFrame. The model might not have followed the strict CSV format.")

        # --- Ensure fallback filename uses the user's input ---
        # The fallback file will now be named RAW_yourfilename.csv
        fallback_filename = f"RAW_{download_filename.replace('.csv', '')}.csv"

        # Fallback to downloading raw text
        st.download_button(
            label="⬇️ Download raw CSV text (for inspection)",
            data=csv_text.encode("utf-8"),
            file_name=fallback_filename,  # Use the dynamic fallback name
            mime="text/csv"
        )
        st.code(csv_text[:1000], language='csv')
        st.error(f"Pandas Parser Error: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred during processing: {e}")

else:
    # Display message if file is missing
    if not uploaded_file:
        st.info("Please upload a PDF file to begin extraction.")
