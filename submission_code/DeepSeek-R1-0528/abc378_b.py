n = int(input().strip())
q_list = []
r_list = []
for _ in range(n):
    q, r = map(int, input().split())
    q_list.append(q)
    r_list.append(r)

Q = int(input().strip())
for _ in range(Q):
    t, d = map(int, input().split())
    q_val = q_list[t-1]
    r_val = r_list[t-1]
    k = (d - r_val + q_val - 1) // q_val
    ans = k * q_val + r_val
    print(ans)