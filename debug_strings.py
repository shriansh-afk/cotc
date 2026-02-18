
def analyze(error):
    error_lower = error.lower()
    print(f"DEBUG: error_lower='{error_lower}'")
    
    if "attributeerror" in error_lower:
        print("DEBUG: attributeerror found")
        if "pypdf2" in error_lower:
            print("DEBUG: pypdf2 found")
        else:
             print("DEBUG: pypdf2 NOT found")
             
        if "pdfreader" in error_lower:
            print("DEBUG: pdfreader found")
        else:
            print("DEBUG: pdfreader NOT found")
            
        if "pypdf2" in error_lower and "pdfreader" in error_lower:
            return "MATCH"
    return "NO MATCH"

err = "AttributeError: module 'PyPDF2' has no attribute 'PdfReader'"
print(f"Result: {analyze(err)}")
