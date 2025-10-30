def main():
    import sys
    input = sys.stdin.readline

    N = int(input())
    A = list(map(int, input().split()))

    # Pair each value with its original index and sort descending by value
    arr = [(v, i) for i, v in enumerate(A)]
    arr.sort(key=lambda x: x[0], reverse=True)

    ans = [0] * N
    current_sum = 0
    i = 0

    # Iterate over groups of equal values in descending order
    while i < N:
        v, _ = arr[i]
        j = i + 1
        # find end of this group (all entries with value v)
        while j < N and arr[j][0] == v:
            j += 1
        # for each index in this group, the sum of greater elements is current_sum
        for k in range(i, j):
            _, idx = arr[k]
            ans[idx] = current_sum
        # update running total by adding this group's total value
        current_sum += v * (j - i)
        i = j

    # Output the answers in the original order
    sys.stdout.write(" ".join(map(str, ans)))


if __name__ == "__main__":
    main()