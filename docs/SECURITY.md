# Security

Uploads are restricted by extension, MIME type, and configurable byte limit. Files are written to a temporary directory with a generated filename and deleted after processing. The service does not execute document content or log document contents, and secrets are supplied through environment variables.

This is not a malware scanner, sandbox, or privacy certification. Deploy behind authentication, TLS, rate limiting, malware scanning, and suitable data retention controls. Do not expose model prompts or private documents through logs or reports.
