n = int(input())
garbages = []
for _ in range(n):
    q, r = map(int, input().split())
    garbages.append((q, r))
q = int(input())
for _ in range(q):
    t, d = map(int, input().split())
    q_val, r_val = garbages[t-1]
    numerator = d - r_val + q_val - 1
    m = numerator // q_val
    next_day = m * q_val + r_val
    print(next_day)