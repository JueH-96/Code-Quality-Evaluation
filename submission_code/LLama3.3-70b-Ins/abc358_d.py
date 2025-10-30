import sys

def solve():
    N, M = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    B = list(map(int, sys.stdin.readline().split()))

    # Sort A in ascending order and keep track of original indices
    sorted_A = sorted((a, i) for i, a in enumerate(A))

    # Sort B in descending order
    sorted_B = sorted((b, i) for i, b in enumerate(B))

    # Initialize variables to keep track of the minimum cost and whether it's possible to satisfy the condition
    min_cost = 0
    possible = True

    # Iterate over the sorted B
    for b, _ in sorted_B:
        # Find the smallest A that is greater than or equal to b
        for a, _ in sorted_A:
            if a >= b:
                min_cost += A[sorted_A[0][1]]
                sorted_A.pop(0)
                break
        else:
            # If no such A is found, it's not possible to satisfy the condition
            possible = False
            break

    # Print the result
    if possible:
        print(min_cost)
    else:
        print(-1)

if __name__ == "__main__":
    solve()