import bisect

def main():
    import sys
    input = sys.stdin.read().split()
    n = int(input[0])
    A = list(map(int, input[1:n+1]))
    sorted_A = sorted(A)
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i+1] = prefix[i] + sorted_A[i]
    total_sum = prefix[-1]
    result = []
    for x in A:
        k = bisect.bisect_right(sorted_A, x)
        res = total_sum - prefix[k]
        result.append(str(res))
    print(' '.join(result))

if __name__ == "__main__":
    main()