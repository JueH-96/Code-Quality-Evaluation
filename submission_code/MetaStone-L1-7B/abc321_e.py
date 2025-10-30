import sys
import math

def compute_case1(X, K, N):
    if K == 0:
        return 1 if X <= N else 0
    left = X * (2 ** K)
    right = left + (2 ** K - 1)
    if right <= N:
        return min(2 ** K, right - left + 1)
    else:
        return max(0, N - left + 1)

def compute_case2(X, K, N):
    max_a = min(K, 20)  # Since 2^20 is over 1e6, sufficient for N up to 1e18
    case2 = 0
    for a in range(1, max_a + 1):
        if a > K:
            continue
        b = K - a
        if b < 0:
            continue
        ancestor = X >> a  # Equivalent to X // (2^a)
        if ancestor == 0:
            continue
        # Compute case1 for ancestor and b steps
        cnt = compute_case1(ancestor, b, N)
        case2 += cnt
    return case2

def main():
    input = sys.stdin.read().split()
    idx = 0
    T = int(input[idx])
    idx += 1
    for _ in range(T):
        N = int(input[idx])
        X = int(input[idx+1])
        K = int(input[idx+2])
        idx +=3
        if K == 0:
            print(1 if X <= N else 0)
            continue
        case1 = compute_case1(X, K, N)
        case2 = compute_case2(X, K, N)
        total = case1 + case2
        print(total)

if __name__ == '__main__':
    main()