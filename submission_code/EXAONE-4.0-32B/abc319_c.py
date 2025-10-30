import math
from collections import deque

def main():
    grid = []
    for _ in range(3):
        row = list(map(int, input().split()))
        grid.append(row)
    
    lines = [
        [(0,0), (0,1), (0,2)],
        [(1,0), (1,1), (1,2)],
        [(2,0), (2,1), (2,2)],
        [(0,0), (1,0), (2,0)],
        [(0,1), (1,1), (2,1)],
        [(0,2), (1,2), (2,2)],
        [(0,0), (1,1), (2,2)],
        [(2,0), (1,1), (0,2)]
    ]
    
    B = []
    for line in lines:
        vals = [grid[i][j] for (i,j) in line]
        if len(set(vals)) == 2:
            from collections import defaultdict
            freq = defaultdict(int)
            for v in vals:
                freq[v] += 1
            once_val = None
            twice_val = None
            for v, cnt in freq.items():
                if cnt == 1:
                    once_val = v
                elif cnt == 2:
                    twice_val = v
            cell1 = None
            cell2 = None
            cell3 = None
            for (i, j) in line:
                if grid[i][j] == twice_val:
                    if cell1 is None:
                        cell1 = (i, j)
                    else:
                        cell2 = (i, j)
                else:
                    cell3 = (i, j)
            B.append((cell1, cell2, cell3))
    
    m = len(B)
    total_permutations = math.factorial(9)
    
    index_map = {}
    for i in range(3):
        for j in range(3):
            idx = i * 3 + j
            index_map[(i, j)] = idx
            
    ans = 0
    for bitmask in range(1 << m):
        edges_set = set()
        for k in range(m):
            if bitmask & (1 << k):
                cell1, cell2, cell3 = B[k]
                u1 = index_map[cell1]
                u2 = index_map[cell2]
                v = index_map[cell3]
                edges_set.add((u1, v))
                edges_set.add((u2, v))
                
        graph = [[] for _ in range(9)]
        pre = [set() for _ in range(9)]
        for (u, v) in edges_set:
            graph[u].append(v)
            pre[v].add(u)
            
        indegree = [len(pre[i]) for i in range(9)]
        q = deque()
        for i in range(9):
            if indegree[i] == 0:
                q.append(i)
                
        count = 0
        while q:
            u = q.popleft()
            count += 1
            for v in graph[u]:
                indegree[v] -= 1
                if indegree[v] == 0:
                    q.append(v)
                    
        if count < 9:
            term = 0
        else:
            dp = [0] * (1 << 9)
            dp[0] = 1
            for mask in range(1 << 9):
                if dp[mask] == 0:
                    continue
                for i in range(9):
                    if mask & (1 << i):
                        continue
                    if pre[i].issubset(set(j for j in range(9) if mask & (1 << j))):
                        new_mask = mask | (1 << i)
                        dp[new_mask] += dp[mask]
            term = dp[(1 << 9) - 1]
            
        sign = 1 if (bin(bitmask).count('1') % 2 == 0) else -1
        ans += sign * term
        
    probability = ans / total_permutations
    print("{:.20f}".format(probability))

if __name__ == "__main__":
    main()