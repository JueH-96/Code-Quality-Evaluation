def solve():
    n, m = map(int, input().split())
    parents = [0] + list(map(int, input().split()))
    insurances = []
    for _ in range(m):
        insurances.append(list(map(int, input().split())))
    
    covered = [False] * (n + 1)
    
    for x, y in insurances:
        covered[x] = True
        
        descendants = find_descendants(x, parents, n, y)
        for desc in descendants:
            covered[desc] = True
            
    count = 0
    for i in range(1, n + 1):
        if covered[i]:
            count += 1
    print(count)

def find_descendants(person, parents, n, generations):
    descendants = []
    
    def dfs(node, gen):
        if gen == 0:
            return
        
        for i in range(2, n + 1):
            if parents[i-1] == node:
                descendants.append(i)
                dfs(i, gen - 1)
    
    dfs(person, generations)
    return descendants

solve()