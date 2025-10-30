def main():
    import sys

    data = sys.stdin.read().strip().split()
    N, X = map(int, data[:2])
    A = list(map(int, data[2:]))

    # Pair each element with its original index (1-based).
    arr = [(A[i], i + 1) for i in range(N)]
    # Sort by the value.
    arr.sort(key=lambda x: x[0])

    # Standard 3-sum approach using two pointers.
    for i in range(N - 2):
        left = i + 1
        right = N - 1
        while left < right:
            s = arr[i][0] + arr[left][0] + arr[right][0]
            if s == X:
                # Retrieve original indices and sort them for output
                i1, i2, i3 = arr[i][1], arr[left][1], arr[right][1]
                ans = sorted([i1, i2, i3])
                print(ans[0], ans[1], ans[2])
                return
            elif s < X:
                left += 1
            else:
                right -= 1

    # If no triple is found
    print(-1)

# Do not forget to call main()
main()