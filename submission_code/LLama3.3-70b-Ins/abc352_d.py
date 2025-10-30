import sys

def solve():
    N, K = map(int, sys.stdin.readline().split())
    P = list(map(int, sys.stdin.readline().split()))

    # Create a dictionary to store the indices of each number in P
    index_dict = {x: i for i, x in enumerate(P)}

    min_diff = float('inf')

    # Iterate over all possible starting numbers for the consecutive sequence
    for start in range(1, N - K + 2):
        # Get the indices of the consecutive sequence
        indices = [index_dict[start + i] for i in range(K)]

        # Sort the indices
        indices.sort()

        # Calculate the difference between the last and first indices
        diff = indices[-1] - indices[0]

        # Update the minimum difference
        min_diff = min(min_diff, diff)

    print(min_diff)

if __name__ == "__main__":
    solve()