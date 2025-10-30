n = int(input())
garbages = []
for _ in range(n):
    q, r = map(int, input().split())
    garbages.append((q, r))
q = int(input())
for _ in range(q):
    t_j, d_j = map(int, input().split())
    q_i, r_i = garbages[t_j - 1]
    numerator = d_j - r_i
    k = (numerator + q_i - 1) // q_i
    x = k * q_i + r_i
    print(x)