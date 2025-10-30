# YOUR CODE HERE
N = int(input())
q_list = []
r_list = []
for _ in range(N):
    q_i, r_i = map(int, input().split())
    q_list.append(q_i)
    r_list.append(r_i)
Q = int(input())
for _ in range(Q):
    t_j, d_j = map(int, input().split())
    t_j -= 1  # Convert to 0-based index
    q = q_list[t_j]
    r = r_list[t_j]
    d_mod_q = d_j % q
    if d_mod_q == r:
        day = d_j
    else:
        delta = (r - d_mod_q + q) % q
        day = d_j + delta
    print(day)