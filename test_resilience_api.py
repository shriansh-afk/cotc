
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from chatbot.executor import DAGExecutor

def test_api_hints():
    executor = DAGExecutor(None, None, None)
    print("Executor initialized.")
    
    # Test 1: PyPDF2 API mismatch
    print("--- Test 1: PyPDF2 ---")
    error_pypdf = "AttributeError: module 'PyPDF2' has no attribute 'PdfReader'"
    hint_pypdf = executor._analyze_dependency_error(error_pypdf)
    print(f"Hint: {hint_pypdf}")
    
    if hint_pypdf and "PyPDF2 API Mismatch" in hint_pypdf:
        print("PASS: PyPDF2 hint")
    else:
        print(f"FAIL: PyPDF2 hint. Got: {hint_pypdf}")

    # Test 2: NameError
    print("\n--- Test 2: NameError ---")
    error_name = "NameError: name 'ebooklib' is not defined"
    hint_name = executor._analyze_dependency_error(error_name)
    print(f"Hint: {hint_name}")
    
    if hint_name and "check your imports" in hint_name.lower():
         print("PASS: NameError hint")
    else:
         print(f"FAIL: NameError hint. Got: {hint_name}")
    
    # Test 3: pdf2epub ImportError
    print("\n--- Test 3: pdf2epub ---")
    error_epub = "ImportError: cannot import name 'convert' from 'pdf2epub'"
    hint_epub = executor._analyze_dependency_error(error_epub)
    print(f"Hint: {hint_epub}")
    
    if hint_epub and "pdf2epub import error" in hint_epub:
        print("PASS: pdf2epub hint")
    else:
        print(f"FAIL: pdf2epub hint. Got: {hint_epub}")

if __name__ == "__main__":
    test_api_hints()
