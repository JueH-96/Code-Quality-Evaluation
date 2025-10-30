import bisect

def main():
    import sys
    input = sys.stdin.read().split()
    n = int(input[0])
    A = list(map(int, input[1:n+1]))
    A.sort()
    sum_A = sum(A)
    S_total = sum_A * (n - 1)
    C = 0
    MOD = 10**8
    for i in range(n):
        T = MOD - A[i]
        j = bisect.bisect_left(A, T, i+1, n)
        C += (n - j)
    ans = S_total - C * MOD
    print(ans)

if __name__ == "__main__":
    main()