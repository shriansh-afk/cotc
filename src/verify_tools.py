
try:
    from chatbot.tools import Tool, PipInstallTool, WebDownloadTool
    print("Successfully imported Tool, PipInstallTool, and WebDownloadTool")
except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"An error occurred: {e}")
