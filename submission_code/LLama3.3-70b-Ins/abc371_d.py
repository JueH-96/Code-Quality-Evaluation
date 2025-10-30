import sys

def solve():
    # Read the number of villages
    N = int(input())

    # Read the coordinates and populations of the villages
    X = list(map(int, input().split()))
    P = list(map(int, input().split()))

    # Read the number of queries
    Q = int(input())

    # Process each query
    for _ in range(Q):
        # Read the query range
        L, R = map(int, input().split())

        # Initialize the total population
        total = 0

        # Iterate over the villages
        for i in range(N):
            # Check if the village is within the query range
            if L <= X[i] <= R:
                # Add the population of the village to the total
                total += P[i]

        # Print the total population
        print(total)

if __name__ == "__main__":
    solve()