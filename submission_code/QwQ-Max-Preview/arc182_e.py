def main():
    import bisect

    N, M, C, K = map(int, input().split())
    A = list(map(int, input().split()))
    A.sort()

    if C == 0:
        min_val = min((a % M) for a in A)
        print(min_val * K)
        return

    sum_total = 0
    for k in range(K):
        x = (C * k) % M
        T = M - x
        # Find the smallest a_i >= T
        idx = bisect.bisect_left(A, T)
        if idx < N:
            a_j = A[idx]
            candidate1 = (x + A[0]) % M
            candidate2 = x + a_j - M
            min_val = min(candidate1, candidate2)
        else:
            min_val = (x + A[0]) % M
        sum_total += min_val

    print(sum_total)

if __name__ == "__main__":
    main()