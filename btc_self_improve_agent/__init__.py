__all__ = ["BTCSelfImprovingAgent", "DataFetchAnalysisAgent", "NewsFetchAnalysisAgent", "ResearchCoordinator"]


def __getattr__(name):
    if name == "BTCSelfImprovingAgent":
        from .agent import BTCSelfImprovingAgent

        return BTCSelfImprovingAgent
    if name in {"DataFetchAnalysisAgent", "NewsFetchAnalysisAgent", "ResearchCoordinator"}:
        from .research_agents import DataFetchAnalysisAgent, NewsFetchAnalysisAgent, ResearchCoordinator

        return {
            "DataFetchAnalysisAgent": DataFetchAnalysisAgent,
            "NewsFetchAnalysisAgent": NewsFetchAnalysisAgent,
            "ResearchCoordinator": ResearchCoordinator,
        }[name]
    raise AttributeError(name)
