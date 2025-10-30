from collections import Counter

def min_operations(N, S, T):
    # Count the frequency of each character in S and T
    s_count = Counter(S)
    t_count = Counter(T)
    
    # Check if it's possible to make S identical to T
    if s_count != t_count:
        return -1
    
    # Find the minimum number of operations required
    operations = 0
    for char in s_count:
        if char != T.index(char):
            operations += 1
    
    return operations

# Read input from stdin
N = int(input())
S = input()
T = input()

# Solve the problem and write output to stdout
print(min_operations(N, S, T))