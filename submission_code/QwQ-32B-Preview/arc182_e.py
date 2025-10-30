import sys
import math

def sum_min_distances(L, R, step, M):
    if L < R:
        gap = R - L
    else:
        gap = (R + M - L) % M
    if gap <= 0:
        return 0
    k_max = (gap - 1) // step
    if k_max < 0:
        return 0
    k_mid = gap // (2 * step)
    sum1 = step * k_mid * (k_mid + 1) // 2
    sum2 = (k_max - k_mid) * gap - step * (k_max * (k_max + 1) // 2 - k_mid * (k_mid + 1) // 2)
    total_sum = sum1 + sum2
    return total_sum

def main():
    N, M, C, K = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    G = math.gcd(C, M)
    P = M // G
    Q = K % P
    A_sorted = sorted(A)
    sum_S = 0
    for i in range(N):
        L = A_sorted[i]
        if i == N - 1:
            R = A_sorted[0] + M
        else:
            R = A_sorted[i + 1]
        sum_S += sum_min_distances(L, R, G, M)
    sum_Q = 0
    for i in range(N):
        L = A_sorted[i]
        if i == N - 1:
            R = A_sorted[0] + M
        else:
            R = A_sorted[i + 1]
        sum_Q += sum_min_distances(L, R, G * Q // P, M)
    result = (K // P) * sum_S + sum_Q
    print(result)

if __name__ == "__main__":
    main()