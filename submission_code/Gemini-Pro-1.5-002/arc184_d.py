def solve():
    n = int(input())
    xy = []
    for _ in range(n):
        x, y = map(int, input().split())
        xy.append((x, y))

    ans = 0
    for i in range(1 << n):
        remaining = []
        for j in range(n):
            if (i >> j) & 1:
                remaining.append(j)

        if not remaining:
            continue

        possible = True
        for j in range(n):
            if (i >> j) & 1:
                temp_remaining = list(remaining)
                
                k = j
                
                removed = []
                for r_idx in temp_remaining:
                    if r_idx != k:
                        if (xy[r_idx][0] < xy[k][0] and xy[r_idx][1] < xy[k][1]) or (xy[r_idx][0] > xy[k][0] and xy[r_idx][1] > xy[k][1]):
                            removed.append(r_idx)
                
                new_remaining = []
                for r_idx in temp_remaining:
                    if r_idx not in removed:
                        new_remaining.append(r_idx)
                
                
                
                
                found = False
                for l in range(1 << n):
                    temp2 = []
                    for m in range(n):
                        if (l >> m) & 1:
                            temp2.append(m)
                    
                    if set(new_remaining) == set(temp2):
                        found = True
                        break
                if not found:
                    possible = False
                    break
        
        if possible:
            ans += 1

    print(ans % 998244353)

solve()