import sys
input = sys.stdin.read

def solve():
    data = input().split()
    index = 0

    N = int(data[index])
    Q = int(data[index + 1])
    index += 2

    R = [int(data[i]) for i in range(index, index + N)]
    index += N

    queries = [int(data[i]) for i in range(index, index + Q)]

    # Create a list of tuples where each tuple is (R[i], i)
    R_with_index = [(R[i], i) for i in range(N)]

    # Sort the list of tuples based on R[i]
    R_with_index.sort()

    # Initialize variables for binary search
    low = 0
    high = N

    # Precompute the prefix sums
    prefix_sums = [0] * (N + 1)
    for i in range(1, N + 1):
        prefix_sums[i] = prefix_sums[i - 1] + R_with_index[i - 1][0]

    results = []

    for X in queries:
        # Binary search to find the maximum number of sleighs
        low = 0
        high = N
        while low < high:
            mid = (low + high + 1) // 2
            if prefix_sums[mid] <= X:
                low = mid
            else:
                high = mid - 1
        results.append(low)

    # Print the results
    for result in results:
        print(result)

if __name__ == "__main__":
    solve()