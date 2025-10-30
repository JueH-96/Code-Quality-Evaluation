import sys

def floor_sum(n, m, a, b):
    """
    Calculates sum_{i=0}^{n-1} floor((a*i + b) / m)
    Assumes a, b >= 0.
    """
    res = 0
    if a == 0:
        return n * (b // m)
    if a >= m:
        res += (n - 1) * n * (a // m) // 2
        a %= m
    if b >= m:
        res += n * (b // m)
        b %= m
    
    y_max = (a * (n - 1) + b) // m
    if y_max == 0:
        return res
    
    x_max = y_max * m - b
    res += (n - (x_max + a - 1) // a) * y_max
    res += floor_sum(y_max, a, m, (a - x_max % a) % a)
    return res

def my_floor_sum(n, m, a, b):
    """
    Wrapper for floor_sum to handle negative b.
    """
    if n == 0:
        return 0
    res = 0
    if b < 0:
        k = (-b + m - 1) // m
        res -= n * k
        b += k * m
    return res + floor_sum(n, m, a, b)

def count_le(n, a, m, y):
    """
    Counts i in [0..n-1] such that (i*a % m) <= y.
    """
    if y < 0:
        return 0
    if y >= m - 1:
        return n
    return my_floor_sum(n, m, a, 0) - my_floor_sum(n, m, a, -y - 1)

def count_ap_mod(n, a, m, l, r):
    """
    Counts i in [0..n-1] such that l < (i*a % m) <= r.
    Handles wrap-around intervals where l > r.
    """
    if n == 0:
        return 0
    if l >= r:
        return count_le(n, a, m, m - 1) - count_le(n, a, m, l) + count_le(n, a, m, r)
    else:
        return count_le(n, a, m, r) - count_le(n, a, m, l)

def sum_ap_mod(n, a, m):
    """
    Calculates sum_{i=0}^{n-1} (i*a % m).
    """
    if n == 0:
        return 0
    a %= m
    return a * n * (n - 1) // 2 - m * my_floor_sum(n, m, a, 0)

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def main():
    N, M, C, K = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))

    if C == 0:
        min_a = A[0]
        for x in A:
            if x < min_a:
                min_a = x
        print(K * min_a)
        return

    A.sort()

    g = gcd(C, M)
    P = M // g

    num_periods = K // P
    rem_len = K % P

    # --- Calculate sum over one full period ---
    # PeriodSum = sum_{j=0}^{P-1} min_i(A_i - jg) mod M
    A_ext = [A[N - 1] - M] + A
    
    period_sum = 0
    for i in range(N):
        start_A = A_ext[i]
        end_A = A_ext[i + 1]
        target = end_A

        j_start = start_A // g + 1
        j_end = end_A // g
        
        n_terms = j_end - j_start + 1
        if n_terms > 0:
            sum_j = n_terms * (j_start + j_end) // 2
            period_sum += n_terms * target - g * sum_j
    
    total_sum = num_periods * period_sum
    
    # --- Calculate sum for the remainder ---
    if rem_len > 0:
        D = (M - C) % M
        
        sum_of_xk = sum_ap_mod(rem_len, D, M)
        
        sum_of_targets = 0

        # Interval [0, A[0]]
        cnt = count_ap_mod(rem_len, D, M, -1, A[0])
        sum_of_targets += A[0] * cnt
        
        # Intervals (A[i], A[i+1]] for i=0..N-2
        for i in range(N - 1):
            cnt = count_ap_mod(rem_len, D, M, A[i], A[i+1])
            sum_of_targets += A[i+1] * cnt

        # Interval (A[N-1], M-1]
        cnt = count_ap_mod(rem_len, D, M, A[N-1], M - 1)
        sum_of_targets += (A[0] + M) * cnt
        
        rem_sum = sum_of_targets - sum_of_xk
        total_sum += rem_sum
        
    print(total_sum)

if __name__ == "__main__":
    main()