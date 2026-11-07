from mcp.server import MCPServer
import agent.tools as tools

def build_server(level: str = "investigator") -> MCPServer:
    if level.lower() not in ["operator", "investigator"]:
        raise ValueError(f"Level {level} is not supported for this tool")

    mcp = MCPServer(level)

    @mcp.tool()
    def search_logs(query: str | None, level: str | None = None, limit: int = 20):
        return tools.search_logs(query, level, limit)

    @mcp.tool()
    def read_file(path: str):
        return tools.read_file(path)

    @mcp.tool()
    def write_file(path: str, content: str):
        return tools.write_file(path, content)

    @mcp.tool()
    def create_handoff(summary: str, evidence: list[str], requested_action: str):
        return tools.create_handoff(summary, evidence, requested_action)
    
    return mcp

mcp = build_server()



if __name__ == "__main__":
    mcp.run()