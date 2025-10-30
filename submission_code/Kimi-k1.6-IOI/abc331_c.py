import bisect

def main():
    import sys
    input = sys.stdin.read().split()
    n = int(input[0])
    a = list(map(int, input[1:n+1]))
    sorted_a = sorted(a)
    # Compute suffix sums
    suffix_sum = [0] * (n + 1)
    for i in range(n-1, -1, -1):
        suffix_sum[i] = sorted_a[i] + suffix_sum[i+1]
    # Prepare results
    result = []
    for num in a:
        idx = bisect.bisect_right(sorted_a, num)
        result.append(str(suffix_sum[idx]))
    print(' '.join(result))

if __name__ == "__main__":
    main()