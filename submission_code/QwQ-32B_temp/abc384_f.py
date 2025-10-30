import sys
from collections import defaultdict

def main():
    data = sys.stdin.read().split()
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    
    if N == 0:
        print(0)
        return
    
    s_max = max(A) * 2
    max_v = 0
    while (1 << max_v) <= s_max:
        max_v += 1
    max_v -= 1  # max_v is the largest exponent where 2^max_v <= s_max

    # Precompute f_val up to s_max
    f_val = [0] * (s_max + 1)
    for s in range(1, s_max + 1):
        if s % 2 == 0:
            f_val[s] = f_val[s // 2]
        else:
            f_val[s] = s

    # Compute T[v] for all v from 0 to max_v
    T = [0] * (max_v + 2)  # T[max_v+1] will be 0

    for v in range(0, max_v + 1):
        m = 1 << v
        count_r = defaultdict(int)
        sum_r = defaultdict(int)
        for a in A:
            r = a % m
            count_r[r] += 1
            sum_r[r] += a
        
        T_v = 0
        for a in A:
            r_a = a % m
            required_r = (m - r_a) % m
            count = count_r.get(required_r, 0)
            s_val = sum_r.get(required_r, 0)
            contribution = a * count + s_val
            T_v += contribution
        T[v] = T_v

    # Compute ordered_sum
    ordered_sum = 0
    for v in range(0, max_v + 1):
        next_v = v + 1
        if next_v > max_v:
            T_next = 0
        else:
            T_next = T[next_v]
        contribution = (T[v] - T_next) // (1 << v)
        ordered_sum += contribution

    # Compute diagonal_sum
    diagonal_sum = 0
    for a in A:
        s = 2 * a
        diagonal_sum += f_val[s]

    # Compute desired_sum
    desired_sum = (ordered_sum + diagonal_sum) // 2
    print(desired_sum)

if __name__ == "__main__":
    main()