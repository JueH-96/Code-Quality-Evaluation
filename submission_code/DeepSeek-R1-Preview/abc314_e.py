n, m = map(int, input().split())
wheels = []
for _ in range(n):
    parts = list(map(int, input().split()))
    c_i = parts[0]
    p_i = parts[1]
    s_list = parts[2:2+p_i]
    wheels.append((c_i, p_i, s_list))

E = [0.0] * m  # E[s] for 0 <= s < m

for s in range(m-1, -1, -1):
    min_e = float('inf')
    for (c_i, p_i, s_list) in wheels:
        k_i = 0
        sum_rest = 0.0
        for s_ij in s_list:
            if s_ij == 0:
                k_i += 1
            else:
                next_s = s + s_ij
                if next_s >= m:
                    contrib = 0.0
                else:
                    contrib = E[next_s]
                sum_rest += contrib
        denominator = p_i - k_i
        if denominator == 0:
            e_i = float('inf')
        else:
            e_i = (c_i * p_i + sum_rest) / denominator
        if e_i < min_e:
            min_e = e_i
    E[s] = min_e

print("{0:.12f}".format(E[0]))