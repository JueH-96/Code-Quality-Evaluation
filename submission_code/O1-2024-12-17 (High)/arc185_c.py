def main():
    import sys
    
    data = sys.stdin.read().strip().split()
    N, X = map(int, data[:2])
    A = list(map(int, data[2:]))

    # Pair each value with its original index (1-based)
    arr = [(A[i], i + 1) for i in range(N)]
    arr.sort(key=lambda x: x[0])  # Sort by the value

    # Standard 3-sum approach using two pointers
    for i in range(N - 2):
        left = i + 1
        right = N - 1
        while left < right:
            current_sum = arr[i][0] + arr[left][0] + arr[right][0]
            if current_sum == X:
                # We found a valid triple; extract original indices
                i1, i2, i3 = arr[i][1], arr[left][1], arr[right][1]
                # Sort the indices to meet the i < j < k requirement
                i1, i2, i3 = sorted([i1, i2, i3])
                print(i1, i2, i3)
                return
            elif current_sum < X:
                left += 1
            else:
                right -= 1

    # If no triple found, print -1
    print(-1)

# Do not forget to call main()
main()