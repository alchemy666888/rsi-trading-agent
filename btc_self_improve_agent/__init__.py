__all__ = ["BTCSelfImprovingAgent"]


def __getattr__(name):
    if name == "BTCSelfImprovingAgent":
        from .agent import BTCSelfImprovingAgent

        return BTCSelfImprovingAgent
    raise AttributeError(name)
