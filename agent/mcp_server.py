from mcp.server import MCPServer

def build_server(level: str = "investigator") -> MCPServer:
    if level.lower() not in ["operator", "investigator"]:
        raise ValueError(f"Level {level} is not supported for this tool")
    return MCPServer("Demo")

mcp = build_server()

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

@mcp.resource("greeting://{name}")
def greeting(name: str) -> str:
    """Greet someone by name"""
    return f"Hello, {name}!"