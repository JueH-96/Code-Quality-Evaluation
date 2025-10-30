def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    maxA = max(A)
    freq = [0] * (maxA + 1)
    for x in A:
        freq[x] += 1

    prefix_sum = [0] * (maxA + 1)
    for i in range(1, maxA + 1):
        prefix_sum[i] = prefix_sum[i - 1] + i * freq[i]

    total_sum = prefix_sum[maxA]
    result = []
    for x in A:
        result.append(str(total_sum - prefix_sum[x]))

    print(" ".join(result))

# Do not forget to call main()
if __name__ == "__main__":
    main()