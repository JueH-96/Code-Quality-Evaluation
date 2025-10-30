# YOUR CODE HERE
import collections

def solve():
    n = int(input())
    edges = []
    graph = collections.defaultdict(list)
    for _ in range(n - 1):
        u, v = map(int, input().split())
        edges.append((u, v))
        graph[u].append(v)
        graph[v].append(u)

    leaves = [i for i in range(1, n + 1) if len(graph[i]) == 1]
    
    ans = []
    removed = set()

    for _ in range(n // 2):
        best_pair = None
        best_dist = -1

        for i in range(len(leaves)):
            for j in range(i + 1, len(leaves)):
                u = leaves[i]
                v = leaves[j]
                if u in removed or v in removed:
                    continue

                q = collections.deque([(u, 0)])
                visited = {u}
                while q:
                    curr, dist = q.popleft()
                    if curr == v:
                        if dist > best_dist:
                            best_dist = dist
                            best_pair = (u, v)
                        break
                    for neighbor in graph[curr]:
                        if neighbor not in visited and neighbor not in removed:
                            visited.add(neighbor)
                            q.append((neighbor, dist + 1))
                            
        ans.append(best_pair)
        removed.add(best_pair[0])
        removed.add(best_pair[1])
        leaves = [i for i in range(1, n + 1) if len(graph[i]) == 1 and i not in removed]

    for pair in ans:
        print(pair[0], pair[1])

solve()