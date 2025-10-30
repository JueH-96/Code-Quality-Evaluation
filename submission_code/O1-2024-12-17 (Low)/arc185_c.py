def main():
    import sys
    data = sys.stdin.read().strip().split()
    N, X = map(int, data[:2])
    A = list(map(int, data[2:]))

    # Create a list of (value, original_index) pairs
    arr = [(A[i], i+1) for i in range(N)]
    # Sort the array by values
    arr.sort(key=lambda x: x[0])

    # Use a 3-sum approach with two pointers
    for i in range(N - 2):
        left = i + 1
        right = N - 1
        while left < right:
            current_sum = arr[i][0] + arr[left][0] + arr[right][0]
            if current_sum == X:
                # Print original indices in ascending order
                indices = sorted([arr[i][1], arr[left][1], arr[right][1]])
                print(indices[0], indices[1], indices[2])
                return
            elif current_sum < X:
                left += 1
            else:
                right -= 1

    # If no triple found
    print(-1)

# Do not forget to call main
main()