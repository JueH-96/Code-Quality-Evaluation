def solve():
    import sys
    data = sys.stdin.read().strip().split()
    N, K = map(int, data[:2])
    A = list(map(int, data[2:]))

    # Sort the array
    A.sort()

    # We need to pick N-K elements such that the difference (max - min) is minimized.
    L = N - K

    # Slide a window of size L over the sorted list and track the minimum difference.
    min_diff = float('inf')
    for i in range(K + 1):
        current_diff = A[i + L - 1] - A[i]
        if current_diff < min_diff:
            min_diff = current_diff

    print(min_diff)

# Call solve after defining it.
solve()