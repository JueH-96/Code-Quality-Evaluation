import bisect

def main():
    import sys
    input = sys.stdin.read().split()
    n = int(input[0])
    A = list(map(int, input[1:n+1]))
    sorted_A = sorted(A)
    prefix_sum = [0]
    for num in sorted_A:
        prefix_sum.append(prefix_sum[-1] + num)
    total_sum = prefix_sum[-1]
    result = []
    for a in A:
        pos = bisect.bisect_right(sorted_A, a)
        sum_greater = total_sum - prefix_sum[pos]
        result.append(str(sum_greater))
    print(' '.join(result))

if __name__ == "__main__":
    main()