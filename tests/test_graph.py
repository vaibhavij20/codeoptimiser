from agent.graph import compiled_graph as app
from agent.state import initial_state


def test_graph():

    state = initial_state(
        """
result=[]
for x in range(100):
    result.append(x*x)
"""
    )

    result = app.invoke(state)

    assert "analysis" in result