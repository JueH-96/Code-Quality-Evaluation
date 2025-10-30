import sys

def solve(N, H):
    """
    For each building i, find the number of integers j (i < j <= N) 
    satisfying the condition: there is no building taller than Building j 
    between Buildings i and j.

    Args:
    N (int): The number of buildings.
    H (list): A list of building heights.

    Returns:
    list: A list of counts for each building.
    """
    counts = [0] * N
    for i in range(N):
        max_height = 0
        for j in range(i + 1, N):
            if H[j] > max_height:
                max_height = H[j]
                counts[i] += 1
    return counts

def main():
    # Read input from stdin
    N = int(input())
    H = list(map(int, input().split()))

    # Solve the problem
    counts = solve(N, H)

    # Write the answer to stdout
    print(*counts)

if __name__ == "__main__":
    main()