import bisect

MOD = 10 ** 8

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:]))
    sum_A = sum(A)
    A.sort()
    count = 0
    for j in range(N):
        target = MOD - A[j]
        left = bisect.bisect_left(A, target, 0, j)
        count += (j - left)
    total = sum_A * (N - 1) - MOD * count
    print(total)

if __name__ == "__main__":
    main()