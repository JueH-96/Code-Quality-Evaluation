class Solution:
    def timeTaken(self, edges: List[List[int]]) -> List[int]:
        n = len(edges) + 1
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        results = []
        for start_node in range(n):
            marked_time = [-1] * n
            marked_time[start_node] = 0
            num_marked = 1
            time = 0

            while num_marked < n:
                time += 1
                newly_marked = []
                for i in range(n):
                    if marked_time[i] == -1:
                        for neighbor in adj[i]:
                            if marked_time[neighbor] != -1:
                                if i % 2 == 1 and marked_time[neighbor] == time - 1:
                                    newly_marked.append(i)
                                    break
                                elif i % 2 == 0 and marked_time[neighbor] == time - 2:
                                    newly_marked.append(i)
                                    break

                for node in newly_marked:
                    if marked_time[node] == -1:
                        marked_time[node] = time
                        num_marked += 1

            results.append(time)

        return results