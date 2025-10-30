import sys

def main():
    N, M = map(int, sys.stdin.readline().split())
    min_at_a = [float('inf')] * (M + 2)  # indexes 0 to M+1

    for _ in range(N):
        L, R = map(int, sys.stdin.readline().split())
        if R < min_at_a[L]:
            min_at_a[L] = R

    # Compute min_b array
    min_b = [0] * (M + 2)
    min_b[M + 1] = float('inf')
    for l in range(M, 0, -1):
        current_min = min(min_at_a[l], min_b[l + 1])
        min_b[l] = current_min

    sum_forbidden = 0
    for l in range(1, M + 1):
        mb = min_b[l]
        if mb == float('inf') or mb > M:
            continue
        cnt = M - mb + 1
        sum_forbidden += cnt

    total = M * (M + 1) // 2
    ans = total - sum_forbidden
    print(ans)

if __name__ == "__main__":
    main()