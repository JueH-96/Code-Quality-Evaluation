# YOUR CODE HERE
import sys

def is_good_pair(N, M, A, B):
    # Create a set to track the pairs (A_i, B_i) and (B_i, A_i)
    pairs = set()
    
    for i in range(M):
        a, b = A[i], B[i]
        if (a, b) in pairs or (b, a) in pairs:
            return "No"
        pairs.add((a, b))
    
    return "Yes"

# Read input
input = sys.stdin.read
N, M, *AB = map(int, input().split())
A = AB[:M]
B = AB[M:]

# Solve and print the result
print(is_good_pair(N, M, A, B))