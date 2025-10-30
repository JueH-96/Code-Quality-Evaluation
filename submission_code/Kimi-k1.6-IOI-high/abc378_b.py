n = int(input())
garbage = [(0, 0)]  # 1-based indexing
for _ in range(n):
    q, r = map(int, input().split())
    garbage.append((q, r))
q_count = int(input())
for _ in range(q_count):
    t, d = map(int, input().split())
    q, r = garbage[t]
    diff = d - r
    if diff <= 0:
        print(r)
    else:
        k = (diff + q - 1) // q
        print(r + k * q)