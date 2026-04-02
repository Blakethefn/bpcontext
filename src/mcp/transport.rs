use anyhow::{Context, Result};
use std::io::{BufRead, BufReader, Read, Write};

/// Read a JSON-RPC message using Content-Length framing (LSP-style)
pub fn read_message(reader: &mut BufReader<impl Read>) -> Result<Option<String>> {
    // Read headers
    let mut content_length: Option<usize> = None;

    loop {
        let mut header_line = String::new();
        let bytes_read = reader.read_line(&mut header_line)
            .context("Failed to read header line")?;

        if bytes_read == 0 {
            return Ok(None); // EOF
        }

        let trimmed = header_line.trim();
        if trimmed.is_empty() {
            break; // End of headers
        }

        if let Some(len_str) = trimmed.strip_prefix("Content-Length:") {
            content_length = Some(
                len_str.trim().parse()
                    .context("Invalid Content-Length value")?
            );
        }
        // Ignore other headers (Content-Type, etc.)
    }

    let length = content_length.context("Missing Content-Length header")?;

    // Read exactly `length` bytes of body
    let mut body = vec![0u8; length];
    reader.read_exact(&mut body)
        .context("Failed to read message body")?;

    let message = String::from_utf8(body)
        .context("Message body is not valid UTF-8")?;

    Ok(Some(message))
}

/// Write a JSON-RPC message with Content-Length framing
pub fn write_message(writer: &mut impl Write, body: &str) -> Result<()> {
    let header = format!("Content-Length: {}\r\n\r\n", body.len());
    writer.write_all(header.as_bytes())?;
    writer.write_all(body.as_bytes())?;
    writer.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_roundtrip() {
        let message = r#"{"jsonrpc":"2.0","id":1,"method":"test"}"#;

        // Write
        let mut buf = Vec::new();
        write_message(&mut buf, message).unwrap();

        // Read back
        let mut reader = BufReader::new(Cursor::new(buf));
        let result = read_message(&mut reader).unwrap().unwrap();
        assert_eq!(result, message);
    }

    #[test]
    fn test_eof_returns_none() {
        let mut reader = BufReader::new(Cursor::new(Vec::<u8>::new()));
        let result = read_message(&mut reader).unwrap();
        assert!(result.is_none());
    }
}
