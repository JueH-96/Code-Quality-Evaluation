def is_good_pair(N, M, A, B):
    # Create a mapping to track pairs
    pairs = {}
    
    for i in range(M):
        a, b = A[i], B[i]
        if a > b:
            a, b = b, a  # Ensure a is always less than or equal to b
        if (a, b) in pairs:
            pairs[(a, b)] += 1
        else:
            pairs[(a, b)] = 1
    
    # If any pair appears more than once, we cannot satisfy the condition
    for count in pairs.values():
        if count > 1:
            return "No"
    
    return "Yes"

import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
A = list(map(int, data[2:M+2]))
B = list(map(int, data[M+2:M+2+M]))

print(is_good_pair(N, M, A, B))