# YOUR CODE HERE
def min_toll(S_x, S_y, T_x, T_y):
    # Calculate the differences in x and y coordinates
    dx = abs(T_x - S_x)
    dy = abs(T_y - S_y)
    
    # The minimum toll is the maximum of the differences divided by 2, plus the parity adjustment
    # The parity adjustment is 1 if the sum of the differences is odd, else 0
    toll = max(dx, dy) // 2
    if (dx + dy) % 2 != 0:
        toll += 1
    
    return toll

# Read input
S_x, S_y = map(int, input().split())
T_x, T_y = map(int, input().split())

# Compute and print the result
print(min_toll(S_x, S_y, T_x, T_y))