def can_reach_end(N, M, A, B, bad_ranges):
    # Create a list to mark bad squares
    bad_squares = [False] * (N + 1)
    
    # Mark bad squares based on the ranges provided
    for L, R in bad_ranges:
        for j in range(L, R + 1):
            if j <= N:
                bad_squares[j] = True
    
    # We start from square 1
    current = 1
    
    # While the current position is less than N
    while current < N:
        # Try to make the largest possible jump that is not landing on a bad square
        found = False
        for jump in range(B, A - 1, -1):
            if current + jump <= N and not bad_squares[current + jump]:
                current += jump
                found = True
                break
        
        # If no valid jump is found, return "No"
        if not found:
            return "No"
    
    # If we reach or exceed N, return "Yes"
    return "Yes"

import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
A = int(data[2])
B = int(data[3])

bad_ranges = []
index = 4
for _ in range(M):
    L = int(data[index])
    R = int(data[index + 1])
    bad_ranges.append((L, R))
    index += 2

result = can_reach_end(N, M, A, B, bad_ranges)
print(result)