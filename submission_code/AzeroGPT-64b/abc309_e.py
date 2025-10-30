from collections import defaultdict

class EulerTour:
    def __init__(self, graph, root=0):
        self.graph = graph
        self.root = root
        self.euler_tour = []
        self.first_visit = {}
        self.last_visit = {}
        self.node_count = 0
        self.dfs(root)

    def dfs(self, node, parent=None):
        self.first_visit[node] = self.node_count
        self.euler_tour.append(node)
        self.node_count += 1

        for child in self.graph[node]:
            if child == parent:
                continue
            self.dfs(child, node)
            self.euler_tour.append(node)
            self.node_count += 1

        self.last_visit[node] = self.node_count - 1

def tree_insurance_coverage(N, M, parents, insurances):
    graph = defaultdict(list)
    for i, parent in enumerate(parents, start=1):
        graph[parent].append(i)

    euler_tour = EulerTour(graph)
    timestamps = [(euler_tour.first_visit[node], euler_tour.last_visit[node]) for node in range(N)]

    covered = set()
    for node, generations in insurances:
        first, last = timestamps[node - 1]
        extend_to = last + generations * 2
        covered.update(euler_tour.euler_tour[first:extend_to + 1])

    return len(covered)

if __name__ == "__main__":
    N, M = map(int, input().split())
    parents = list(map(int, input().split()))
    insurances = [tuple(map(int, input().split())) for _ in range(M)]

    result = tree_insurance_coverage(N, M, parents, insurances)
    print(result)