use async_openai::{Client, config::OpenAIConfig};
use clap::Parser;
use serde_json::{Value, json};
use std::{env, fs, path::Path, process};

#[derive(Parser)]
#[command(author, version, about)]
struct Args {
    #[arg(short = 'p', long)]
    prompt: String,
}

fn execute_tool(name: &str, arguments: &Value) -> String {
    match name {
        "Read" => {
            let file_path = arguments["file_path"].as_str().unwrap_or("");
            match fs::read_to_string(file_path) {
                Ok(content) => content,
                Err(e) => format!("Error reading file: {}", e),
            }
        }
        "Write" => {
            let file_path = arguments["file_path"].as_str().unwrap_or("");
            let content = arguments["content"].as_str().unwrap_or("");
            // Create parent directories if needed
            if let Some(parent) = Path::new(file_path).parent() {
                if !parent.as_os_str().is_empty() {
                    let _ = fs::create_dir_all(parent);
                }
            }
            match fs::write(file_path, content) {
                Ok(()) => format!("Successfully wrote to {}", file_path),
                Err(e) => format!("Error writing file: {}", e),
            }
        }
        "Bash" => {
            let command = arguments["command"].as_str().unwrap_or("");
            match std::process::Command::new("/bin/sh")
                .arg("-c")
                .arg(command)
                .output()
            {
                Ok(output) => {
                    let stdout = String::from_utf8_lossy(&output.stdout);
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    let mut result = String::new();
                    if !stdout.is_empty() {
                        result.push_str(&stdout);
                    }
                    if !stderr.is_empty() {
                        if !result.is_empty() {
                            result.push('\n');
                        }
                        result.push_str(&stderr);
                    }
                    if result.is_empty() {
                        "Command executed successfully (no output)".to_string()
                    } else {
                        result
                    }
                }
                Err(e) => format!("Error executing command: {}", e),
            }
        }
        _ => format!("Unknown tool: {}", name),
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let base_url = env::var("OPENROUTER_BASE_URL")
        .unwrap_or_else(|_| "https://openrouter.ai/api/v1".to_string());

    let api_key = env::var("OPENROUTER_API_KEY").unwrap_or_else(|_| {
        eprintln!("OPENROUTER_API_KEY is not set");
        process::exit(1);
    });

    let config = OpenAIConfig::new()
        .with_api_base(base_url)
        .with_api_key(api_key);

    let client = Client::with_config(config);

    let tools = json!([
        {
            "type": "function",
            "function": {
                "name": "Read",
                "description": "Read and return the contents of a file",
                "parameters": {
                    "type": "object",
                    "required": ["file_path"],
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "The path to the file to read"
                        }
                    }
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "Write",
                "description": "Write content to a file",
                "parameters": {
                    "type": "object",
                    "required": ["file_path", "content"],
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "The path of the file to write to"
                        },
                        "content": {
                            "type": "string",
                            "description": "The content to write to the file"
                        }
                    }
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "Bash",
                "description": "Execute a shell command",
                "parameters": {
                    "type": "object",
                    "required": ["command"],
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The command to execute"
                        }
                    }
                }
            }
        }
    ]);

    let mut messages = vec![
        json!({
            "role": "user",
            "content": args.prompt
        })
    ];

    // Agent loop
    loop {
        let response: Value = client
            .chat()
            .create_byot(json!({
                "messages": messages,
                "model": "anthropic/claude-haiku-4.5",
                "tools": tools,
            }))
            .await?;

        let choice = &response["choices"][0];
        let message = &choice["message"];

        // Append assistant message to conversation
        messages.push(message.clone());

        // Check for tool calls
        if let Some(tool_calls) = message["tool_calls"].as_array() {
            if !tool_calls.is_empty() {
                for tool_call in tool_calls {
                    let name = tool_call["function"]["name"].as_str().unwrap_or("");
                    let arguments: Value = match tool_call["function"]["arguments"].as_str() {
                        Some(args_str) => serde_json::from_str(args_str).unwrap_or(json!({})),
                        None => tool_call["function"]["arguments"].clone(),
                    };
                    let id = tool_call["id"].as_str().unwrap_or("");

                    let result = execute_tool(name, &arguments);

                    messages.push(json!({
                        "role": "tool",
                        "tool_call_id": id,
                        "content": result
                    }));
                }
                // Continue loop to send results back to LLM
                continue;
            }
        }

        // No tool calls - output final response and exit
        // You can use print statements as follows for debugging, they'll be visible when running tests.
        eprintln!("Logs from your program will appear here!");

        if let Some(content) = message["content"].as_str() {
            println!("{}", content);
        }

        break;
    }

    Ok(())
}
