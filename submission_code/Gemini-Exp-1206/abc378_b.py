def solve():
    n = int(input())
    q_values = []
    r_values = []
    for _ in range(n):
        q, r = map(int, input().split())
        q_values.append(q)
        r_values.append(r)
    
    q_num = int(input())
    for _ in range(q_num):
        t, d = map(int, input().split())
        t -= 1
        
        if d % q_values[t] == r_values[t]:
            print(d)
        else:
            next_day = (d // q_values[t]) * q_values[t] + r_values[t]
            if next_day < d:
                next_day += q_values[t]
            print(next_day)

solve()