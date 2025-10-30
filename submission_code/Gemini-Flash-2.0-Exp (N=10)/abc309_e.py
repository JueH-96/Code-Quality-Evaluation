def solve():
    n, m = map(int, input().split())
    if n == 1:
        print(1 if m > 0 else 0)
        return
    parents = [0] + list(map(int, input().split()))
    
    covered = [False] * (n + 1)
    
    for _ in range(m):
        x, y = map(int, input().split())
        
        q = [(x, 0)]
        visited = set()
        
        while q:
            curr_person, curr_gen = q.pop(0)
            if curr_person not in visited:
                visited.add(curr_person)
                covered[curr_person] = True
                if curr_gen < y:
                    for i in range(2, n + 1):
                        if parents[i-1] == curr_person:
                            q.append((i, curr_gen + 1))
    
    print(sum(covered[1:]))

solve()