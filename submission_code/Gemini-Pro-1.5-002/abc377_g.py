def solve():
    n = int(input())
    s = [input() for _ in range(n)]

    for i in range(n):
        t = s[i]
        min_cost = len(t)  # Cost to make T empty

        for j in range(i):
            u = s[j]
            lcs_len = 0
            for k in range(min(len(t), len(u))):
                if t[k] == u[k]:
                    lcs_len += 1
                else:
                    break
            
            cost = (len(t) - lcs_len) + (len(u) - lcs_len)
            min_cost = min(min_cost, cost)
        
        print(min_cost)

solve()