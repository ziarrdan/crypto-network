class Graph:
    def __init__(self, graph, start, end):
        self.graph = graph
        self.start = start
        self.end = end

    def get_graph(self):
        return self.graph

    def get_dates(self):
        return self.start, self.end
