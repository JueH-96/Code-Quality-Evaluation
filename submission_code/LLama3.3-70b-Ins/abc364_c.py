import sys
import itertools

def solve(N, X, Y, A, B):
    # Generate all permutations of dishes
    perms = list(itertools.permutations(range(N)))

    # Initialize minimum number of dishes
    min_dishes = float('inf')

    # Iterate over all permutations
    for perm in perms:
        # Initialize total sweetness and saltiness
        total_sweetness = 0
        total_saltiness = 0

        # Initialize number of dishes eaten
        num_dishes = 0

        # Iterate over dishes in permutation
        for i in perm:
            # Add sweetness and saltiness of current dish
            total_sweetness += A[i]
            total_saltiness += B[i]

            # Increment number of dishes eaten
            num_dishes += 1

            # Check if total sweetness or saltiness exceeds limit
            if total_sweetness > X or total_saltiness > Y:
                break

        # Update minimum number of dishes
        min_dishes = min(min_dishes, num_dishes)

    return min_dishes

def main():
    # Read input from stdin
    N, X, Y = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    B = list(map(int, sys.stdin.readline().split()))

    # Solve problem
    result = solve(N, X, Y, A, B)

    # Write result to stdout
    print(result)

if __name__ == "__main__":
    main()