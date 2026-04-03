use anyhow::{Context, Result};
use std::io::{BufRead, BufReader, Read, Write};

/// Read a JSON-RPC message, auto-detecting the framing format.
///
/// Supports two formats:
/// - Newline-delimited JSON (used by Claude Code): `{...}\n`
/// - Content-Length framing (LSP-style): `Content-Length: N\r\n\r\n{...}`
pub fn read_message(reader: &mut BufReader<impl Read>) -> Result<Option<String>> {
    loop {
        let mut line = String::new();
        let bytes_read = reader.read_line(&mut line).context("Failed to read line")?;

        if bytes_read == 0 {
            return Ok(None); // EOF
        }

        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue; // Skip blank lines
        }

        // If the line starts with '{', it's newline-delimited JSON
        if trimmed.starts_with('{') {
            return Ok(Some(trimmed.to_string()));
        }

        // Otherwise treat as Content-Length header (LSP-style framing)
        if let Some(len_str) = trimmed.strip_prefix("Content-Length:") {
            let length: usize = len_str
                .trim()
                .parse()
                .context("Invalid Content-Length value")?;

            // Consume remaining headers until empty line
            loop {
                let mut header = String::new();
                reader
                    .read_line(&mut header)
                    .context("Failed to read header")?;
                if header.trim().is_empty() {
                    break;
                }
            }

            // Read exactly `length` bytes of body
            let mut body = vec![0u8; length];
            reader
                .read_exact(&mut body)
                .context("Failed to read message body")?;

            let message = String::from_utf8(body).context("Message body is not valid UTF-8")?;

            return Ok(Some(message));
        }

        // Skip unrecognized header lines
    }
}

/// Write a JSON-RPC message as newline-delimited JSON
pub fn write_message(writer: &mut impl Write, body: &str) -> Result<()> {
    writer.write_all(body.as_bytes())?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_newline_delimited_roundtrip() {
        let message = r#"{"jsonrpc":"2.0","id":1,"method":"test"}"#;

        // Write (newline-delimited)
        let mut buf = Vec::new();
        write_message(&mut buf, message).unwrap();

        // Read back
        let mut reader = BufReader::new(Cursor::new(buf));
        let result = read_message(&mut reader).unwrap().unwrap();
        assert_eq!(result, message);
    }

    #[test]
    fn test_content_length_framing() {
        let message = r#"{"jsonrpc":"2.0","id":1,"method":"test"}"#;
        let framed = format!("Content-Length: {}\r\n\r\n{}", message.len(), message);

        let mut reader = BufReader::new(Cursor::new(framed.into_bytes()));
        let result = read_message(&mut reader).unwrap().unwrap();
        assert_eq!(result, message);
    }

    #[test]
    fn test_eof_returns_none() {
        let mut reader = BufReader::new(Cursor::new(Vec::<u8>::new()));
        let result = read_message(&mut reader).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_blank_lines_skipped() {
        let input = "\n\n{\"jsonrpc\":\"2.0\",\"id\":1}\n";
        let mut reader = BufReader::new(Cursor::new(input.as_bytes().to_vec()));
        let result = read_message(&mut reader).unwrap().unwrap();
        assert_eq!(result, r#"{"jsonrpc":"2.0","id":1}"#);
    }
}
