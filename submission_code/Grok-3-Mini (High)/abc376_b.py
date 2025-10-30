import sys

# Read all input and split into a list
data = sys.stdin.read().split()
index = 0

# Read N and Q
N = int(data[index])
index += 1
Q = int(data[index])
index += 1

# Define the distance function to compute shortest path avoiding F
def dist_avoid(C, T, F, N):
    D_cw = (T - C) % N
    D_ccw = (C - T) % N
    D_short = min(D_cw, D_ccw)
    D_long = max(D_cw, D_ccw)
    if D_short == D_long:
        return D_short
    else:
        if D_cw == D_short:  # short path is clockwise
            if 0 <= (F - C) % N <= D_cw:
                return D_long
            else:
                return D_short
        else:  # short path is counterclockwise
            if 0 <= (C - F) % N <= D_ccw:
                return D_long
            else:
                return D_short

# Initialize positions and total moves
left_pos = 1
right_pos = 2
total_moves = 0

# Process each query
for _ in range(Q):
    H_str = data[index]  # "L" or "R"
    T_pos = int(data[index + 1])  # Target position
    index += 2  # Move index forward by 2
    
    if H_str == "L":
        # Move left hand to T_pos, right hand fixed
        C = left_pos
        target = T_pos
        F = right_pos  # Forbidden position
        dist = dist_avoid(C, target, F, N)
        total_moves += dist
        left_pos = target  # Update left position
    elif H_str == "R":
        # Move right hand to T_pos, left hand fixed
        C = right_pos
        target = T_pos
        F = left_pos  # Forbidden position
        dist = dist_avoid(C, target, F, N)
        total_moves += dist
        right_pos = target  # Update right position

# Output the total number of moves
print(total_moves)