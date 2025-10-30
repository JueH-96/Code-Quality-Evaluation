import bisect

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:N+1]))
    A.sort()
    n = len(A)
    prefix_sum = [0] * (n + 1)
    for i in range(n):
        prefix_sum[i+1] = prefix_sum[i] + A[i]
    total = 0
    for i in range(n):
        x = A[i]
        if i == n-1:
            break
        if x == 1:
            sum_i = prefix_sum[n] - prefix_sum[i+1]
        else:
            sum_i = 0
            max_k = A[-1] // x
            for k in range(1, max_k + 1):
                target = k * x
                idx = bisect.bisect_left(A, target, i+1, n)
                count = n - idx
                sum_i += count
        total += sum_i
    print(total)

if __name__ == '__main__':
    main()