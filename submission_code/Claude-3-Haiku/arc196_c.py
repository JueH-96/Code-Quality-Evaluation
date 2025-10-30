import sys

def count_strongly_connected_partitions(N, S):
    MOD = 998244353
    
    # Count the number of white and black vertices
    white_count = S.count('W')
    black_count = S.count('B')
    
    # Check if the number of white and black vertices is equal
    if white_count != black_count:
        return 0
    
    # Compute the number of ways to partition the vertices into pairs
    result = 1
    for i in range(1, N+1):
        result = (result * i) % MOD
    
    return result

# Read the input
N = int(input())
S = input()

# Compute the answer and print it
print(count_strongly_connected_partitions(N, S))