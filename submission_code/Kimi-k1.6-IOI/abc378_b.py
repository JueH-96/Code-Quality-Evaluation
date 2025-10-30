n = int(input())
garbage = [None]  # 1-based indexing
for _ in range(n):
    q, r = map(int, input().split())
    garbage.append((q, r))
q = int(input())
for _ in range(q):
    t, d = map(int, input().split())
    q_t, r_t = garbage[t]
    rem = d % q_t
    if rem == r_t:
        print(d)
    else:
        if rem < r_t:
            ans = d + (r_t - rem)
        else:
            ans = d + (q_t - (rem - r_t))
        print(ans)