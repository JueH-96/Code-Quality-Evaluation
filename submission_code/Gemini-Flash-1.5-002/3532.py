class Solution:
    def timeTaken(self, edges: List[List[int]]) -> List[int]:
        n = len(edges) + 1
        graph = [[] for _ in range(n)]
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)

        times = []
        for start_node in range(n):
            marked = [False] * n
            marked_times = [-1] * n
            queue = [(start_node, 0)]
            marked[start_node] = True
            marked_times[start_node] = 0
            max_time = 0

            while queue:
                node, time = queue.pop(0)
                max_time = max(max_time, time)

                for neighbor in graph[node]:
                    if not marked[neighbor]:
                        mark_time = -1
                        if neighbor % 2 != 0:
                            mark_time = time + 1
                        else:
                            mark_time = time + 2
                        marked[neighbor] = True
                        marked_times[neighbor] = mark_time
                        queue.append((neighbor, mark_time))

            times.append(max(marked_times))

        return times