import sys

def solve():
    N = int(sys.stdin.readline())
    S = sys.stdin.readline().strip()

    # Step 1: Find all 0-indexed positions of '1's.
    p_indices = []
    for i in range(N):
        if S[i] == '1':
            p_indices.append(i)

    k = len(p_indices)

    # If there is only one '1', it is already contiguous.
    # The problem guarantees at least one '1', so k >= 1.
    if k <= 1:
        print(0)
        return

    # Step 2: Calculate a_i = p_i - i for each '1'.
    # p_indices are naturally sorted (as we iterate through S).
    # This also ensures a_values will be sorted.
    a_values = [p_indices[i] - i for i in range(k)]

    # Step 3: Find the median of a_values.
    # For a sorted list of k elements, the median is at index k // 2.
    median_a = a_values[k // 2]

    # Step 4: Calculate the total number of operations.
    # This is the sum of absolute differences between each a_i and the median.
    total_operations = 0
    for val in a_values:
        total_operations += abs(val - median_a)

    print(total_operations)

if __name__ == '__main__':
    solve()