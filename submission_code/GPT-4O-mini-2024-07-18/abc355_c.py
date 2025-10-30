def bingo_turn(N, T, A):
    marked_rows = [0] * N
    marked_cols = [0] * N
    marked_diag1 = 0  # top-left to bottom-right
    marked_diag2 = 0  # top-right to bottom-left
    
    # Create a mapping from number to its position in the grid
    position = {}
    for i in range(N):
        for j in range(N):
            position[N * i + j + 1] = (i, j)
    
    for turn in range(T):
        number = A[turn]
        if number in position:
            i, j = position[number]
            marked_rows[i] += 1
            marked_cols[j] += 1
            
            if i == j:
                marked_diag1 += 1
            if i + j == N - 1:
                marked_diag2 += 1
            
            # Check for Bingo
            if (marked_rows[i] == N or 
                marked_cols[j] == N or 
                marked_diag1 == N or 
                marked_diag2 == N):
                return turn + 1  # return the turn number (1-indexed)
    
    return -1  # No Bingo achieved

import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
T = int(data[1])
A = list(map(int, data[2:T+2]))

result = bingo_turn(N, T, A)
print(result)