import sys, bisect

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    A.sort()
    sum_A = sum(A)
    count = 0
    target = 100_000_000
    for i in range(N-1):
        req = target - A[i]
        j = bisect.bisect_left(A, req, i+1, N)
        count += N - j
    result = sum_A * (N-1) - count * target
    print(result)

if __name__ == "__main__":
    main()