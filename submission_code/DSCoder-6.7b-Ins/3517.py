class Solution:
    def shortestDistanceAfterQueries(self, n: int, queries: List[List[int]]) -> List[int]:
        parent = list(range(n))
        size = [1] * n
        answer = []

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            x, y = find(x), find(y)
            if x == y:
                return
            if size[x] < size[y]:
                x, y = y, x
            parent[y] = x
            size[x] += size[y]

        for q in queries:
            x, y = q[0], q[1]
            union(x, y)
            answer.append(size[find(0)])
        return answer