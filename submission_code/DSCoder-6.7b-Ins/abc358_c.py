import sys
from collections import defaultdict

def main():
    N, M = map(int, sys.stdin.readline().split())
    S = [sys.stdin.readline().strip() for _ in range(N)]

    # Create a dictionary to store the stands that sell each flavor
    flavor_stands = defaultdict(list)
    for i in range(N):
        for j in range(M):
            if S[i][j] == 'o':
                flavor_stands[j+1].append(i+1)

    # Sort the stands for each flavor by their number
    for key in flavor_stands.keys():
        flavor_stands[key].sort()

    # Initialize the minimum number of stands and the current stand
    min_stands = N + 1
    current_stand = 1

    # Iterate over the flavors
    for flavor in sorted(flavor_stands.keys()):
        # Find the next stand that is the maximum of the current stand and the minimum of the stands that sell this flavor
        next_stand = max(current_stand, min(flavor_stands[flavor]))
        # Update the minimum number of stands and the current stand
        min_stands = min(min_stands, next_stand)
        current_stand = next_stand

    # Print the minimum number of stands
    print(min_stands)

if __name__ == "__main__":
    main()