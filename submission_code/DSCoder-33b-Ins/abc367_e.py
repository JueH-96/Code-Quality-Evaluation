# YOUR CODE HERE

import sys

def solve(n, k, x, a):
    # Create a list of tuples (x[i], a[i])
    pairs = list(zip(x, a))
    # Sort the list of tuples by the first element of each tuple
    pairs.sort()
    # Create a list of the second elements of each tuple
    b = [pair[1] for pair in pairs]
    # Perform the operation K times
    for _ in range(k):
        # Replace A with B such that B_i = A_{X_i}
        a = [b[i] for i in a]
    return a

def main():
    # Read the input
    n, k = map(int, sys.stdin.readline().split())
    x = list(map(int, sys.stdin.readline().split()))
    a = list(map(int, sys.stdin.readline().split()))
    # Solve the problem
    a = solve(n, k, x, a)
    # Print the result
    print(' '.join(map(str, a)))

if __name__ == '__main__':
    main()