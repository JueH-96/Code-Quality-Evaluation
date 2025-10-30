def solve():
    n, m = map(int, input().split())
    parents = [0] + list(map(int, input().split()))
    children = [[] for _ in range(n + 1)]
    for i in range(2, n + 1):
        children[parents[i-1]].append(i)
    
    insurance = []
    for _ in range(m):
        insurance.append(list(map(int, input().split())))
    
    covered = [False] * (n + 1)
    
    for x, y in insurance:
        q = [(x, 0)]
        covered[x] = True
        
        while q:
            curr, gen = q.pop(0)
            
            if gen < y:
                for child in children[curr]:
                    covered[child] = True
                    q.append((child, gen + 1))
    
    print(sum(covered))

solve()