import math
from collections import defaultdict

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def solve(N, K, A):
    # Create a dictionary to store the maximum GCD for each element
    max_gcd = defaultdict(int)

    for i in range(N):
        # Initialize the maximum GCD to 0
        max_gcd[i+1] = 0

        # Iterate through all possible combinations of K elements that include A[i]
        for j in range(1, 2**K):
            # Check if the current combination includes A[i]
            if j & (1 << (K-1)):
                # Calculate the GCD of the current combination
                curr_gcd = 0
                for k in range(K):
                    if j & (1 << k):
                        curr_gcd = gcd(curr_gcd, A[i+k])
                # Update the maximum GCD for A[i]
                max_gcd[i+1] = max(max_gcd[i+1], curr_gcd)

    # Print the answers
    for i in range(1, N+1):
        print(max_gcd[i])

# Read the input
N, K = map(int, input().split())
A = list(map(int, input().split()))

# Solve the problem
solve(N, K, A)