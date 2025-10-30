import sys
import math

def ceil_div(n, d):
    if d == 0:
        if n == 0:
            return 0
        else:
            raise ValueError("Division by zero")
    q = n // d
    r = n % d
    if r == 0:
        return q
    else:
        return q + 1 if n * d > 0 else q

def check(s, N, Q, A, B):
    lower = -1
    upper = math.inf
    for i in range(N):
        diff = A[i] - B[i]
        if diff > 0:
            if B[i] * s > Q[i]:
                return False
            ub = (Q[i] - s * B[i]) // diff
            if upper == math.inf:
                upper = ub
            else:
                upper = min(upper, ub)
        elif diff < 0:
            if B[i] * s > Q[i]:
                return False
            lb = ceil_div(Q[i] - s * B[i], diff)
            if lower == -1:
                lower = lb
            else:
                lower = max(lower, lb)
        else:
            if B_i != 0 and s * B[i] > Q[i]:
                return False
    if lower == -1:
        lower = 0
    if upper == math.inf:
        upper = s
    lower = max(lower, 0)
    upper = min(upper, s)
    return lower <= upper

def main():
    import sys
    N = int(sys.stdin.readline())
    Q = list(map(int, sys.stdin.readline().split()))
    A = list(map(int, sys.stdin.readline().split()))
    B = list(map(int, sys.stdin.readline().split()))
    
    # Compute max_a and max_b
    max_a = float('inf')
    for i in range(N):
        if A[i] > 0:
            max_a = min(max_a, Q[i] // A[i])
    if max_a == float('inf'):
        max_a = 0
    
    max_b = float('inf')
    for i in range(N):
        if B[i] > 0:
            max_b = min(max_b, Q[i] // B[i])
    if max_b == float('inf'):
        max_b = 0
    
    low = 0
    high = max_a + max_b
    
    while low < high:
        mid = (low + high + 1) // 2
        if check(mid, N, Q, A, B):
            low = mid
        else:
            high = mid - 1
    print(low)

if __name__ == "__main__":
    main()