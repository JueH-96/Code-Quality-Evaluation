def solve():
    n, m = map(int, input().split())
    stronger = [[] for _ in range(n + 1)]
    for _ in range(m):
        a, b = map(int, input().split())
        stronger[a].append(b)

    def is_stronger(x, y, known_stronger):
        if x == y:
            return False
        
        q = [x]
        visited = {x}
        
        while q:
            curr = q.pop(0)
            if curr == y:
                return True
            
            for neighbor in known_stronger[curr]:
                if neighbor not in visited:
                    q.append(neighbor)
                    visited.add(neighbor)
        return False

    possible_strongest = []
    for i in range(1, n + 1):
        is_strongest = True
        for j in range(1, n + 1):
            if i != j:
                if not is_stronger(i, j, stronger):
                    is_strongest = False
                    break
        if is_strongest:
            possible_strongest.append(i)
    
    if len(possible_strongest) == 1:
        print(possible_strongest[0])
    else:
        print("-1")

solve()