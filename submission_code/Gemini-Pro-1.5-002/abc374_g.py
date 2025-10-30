def solve():
    n = int(input())
    s = [input() for _ in range(n)]
    
    adj = [[] for _ in range(26)]
    indeg = [0] * 26
    outdeg = [0] * 26
    used = [False] * 26
    
    for si in s:
        u = ord(si[0]) - ord('A')
        v = ord(si[1]) - ord('A')
        adj[u].append(v)
        indeg[v] += 1
        outdeg[u] += 1
        used[u] = True
        used[v] = True
    
    ans = 0
    for i in range(26):
        if used[i] and indeg[i] == 0:
            ans += 1
            curr = i
            while True:
                found = False
                for neighbor in adj[curr]:
                    if indeg[neighbor] == 1:
                        indeg[neighbor] -=1
                        outdeg[curr] -= 1
                        curr = neighbor
                        found = True
                        break
                if not found:
                    break

    return ans

print(solve())