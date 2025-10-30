def solve():
    n, m = map(int, input().split())
    stronger = {}
    for _ in range(m):
        a, b = map(int, input().split())
        if a not in stronger:
            stronger[a] = set()
        stronger[a].add(b)
    
    def is_stronger(x, y, current_stronger):
        if x == y:
            return False
        
        q = [x]
        visited = {x}
        while q:
            curr = q.pop(0)
            if curr in current_stronger:
                for next_person in current_stronger[curr]:
                    if next_person == y:
                        return True
                    if next_person not in visited:
                        visited.add(next_person)
                        q.append(next_person)
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
        print(-1)

solve()