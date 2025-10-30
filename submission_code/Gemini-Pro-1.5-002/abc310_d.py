# YOUR CODE HERE
def solve():
    n, t, m = map(int, input().split())
    incompatible = []
    for _ in range(m):
        u, v = map(int, input().split())
        incompatible.append((u - 1, v - 1))

    ans = 0
    for i in range(t**n):
        teams = [[] for _ in range(t)]
        temp = i
        for j in range(n):
            teams[temp % t].append(j)
            temp //= t
        
        valid = True
        for team in teams:
            if not team:
                valid = False
                break

        if not valid:
            continue
            
        for u, v in incompatible:
            found = False
            for team in teams:
                if u in team and v in team:
                    found = True
                    break
            if found:
                valid = False
                break
        
        if valid:
            ans += 1

    print(ans)

solve()