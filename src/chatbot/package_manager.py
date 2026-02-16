import subprocess
import sys
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

@dataclass
class PackageInfo:
    name: str
    installed: bool = False
    failed_attempts: int = 0
    last_error: Optional[str] = None

class PackageManager:
    def __init__(self, max_retries: int = 3):
        self.packages: Dict[str, PackageInfo] = {}
        self.max_retries = max_retries

    def _parse_package_name(self, package_spec: str) -> tuple[str, str | None]:
        """Parse package name and version specifier.
        
        Examples:
            "numpy==1.21.0" -> ("numpy", "==1.21.0")
            "requests>=2.0" -> ("requests", ">=2.0")
            "pandas" -> ("pandas", None)
        """
        import re
        # Match package name (alphanumeric, -, _, .) and optional version specifiers
        match = re.match(r'^([a-zA-Z0-9_\-\.]+)(.*)$', package_spec)
        if match:
            return match.group(1).lower(), match.group(2) if match.group(2) else None
        return package_spec.lower(), None

    def install(self, packages: List[str], upgrade: bool = False) -> dict:
        """Smart installation of packages with loop prevention and version fallback.
        
        Returns a dict with success status, message, output, and error.
        """
        to_install = []
        messages = []

        for pkg_spec in packages:
            name, version_spec = self._parse_package_name(pkg_spec)
            
            if name not in self.packages:
                self.packages[name] = PackageInfo(name=name)
            
            info = self.packages[name]
            
            # 1. Loop Prevention: Check if we've failed too many times
            if info.failed_attempts >= self.max_retries:
                msg = f"Skipping '{pkg_spec}': Failed {info.failed_attempts} times previously. Please check compatibility."
                messages.append(msg)
                logger.warning(msg)
                continue
            
            # 2. Smart Version Fallback: If previous attempts failed, try without version
            final_spec = pkg_spec
            if info.failed_attempts > 0 and version_spec:
                final_spec = name
                msg = f"Retrying '{name}' without version constraints (attempt {info.failed_attempts + 1})"
                messages.append(msg)
                logger.info(msg)
            
            # 3. Avoid redundant installs? 
            # If already installed, we might skip, but 'upgrade=True' or re-install request might be intentional.
            # For now, we allow re-install attempts unless max_retries reached.
            
            to_install.append(final_spec)

        if not to_install:
            return {
                "success": False, 
                "message": "\n".join(messages) or "No valid packages to install (all skipped due to failures).",
                "skipped": True
            }

        # Construct pip command
        cmd = [sys.executable, "-m", "pip", "install"]
        if upgrade:
            cmd.append("--upgrade")
        cmd.extend(to_install)
        
        logger.info(f"Running pip install: {' '.join(cmd)}")

        try:
            # 5-minute timeout for installation
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                # Success
                for spec in to_install:
                    name, _ = self._parse_package_name(spec)
                    # Mark as installed and reset failures? 
                    # If we successfully installed 'numpy' after failing 'numpy==1.21.0', 
                    # we should count it as success.
                    self.packages[name].installed = True
                    self.packages[name].failed_attempts = 0 
                
                return {
                    "success": True,
                    "message": f"Successfully installed: {', '.join(to_install)}\n" + "\n".join(messages),
                    "stdout": result.stdout
                }
            else:
                # Failure
                for spec in to_install:
                    name, _ = self._parse_package_name(spec)
                    self.packages[name].failed_attempts += 1
                    self.packages[name].last_error = result.stderr
                
                logger.error(f"Pip install failed: {result.stderr}")
                
                return {
                    "success": False,
                    "error": f"Pip install failed with code {result.returncode}:\n{result.stderr}\n" + "\n".join(messages)
                }
                
        except subprocess.TimeoutExpired:
            return {
                "success": False, 
                "error": "Pip install timed out after 300 seconds."
            }
        except Exception as e:
            return {"success": False, "error": f"Unexpected error: {str(e)}"}
