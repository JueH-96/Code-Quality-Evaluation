# YOUR CODE HERE
def min_toll(S_x, S_y, T_x, T_y):
    # Calculate the differences in x and y coordinates
    dx = abs(T_x - S_x)
    dy = abs(T_y - S_y)
    
    # Determine the parity of the starting and ending positions
    start_parity = (S_x + S_y) % 2
    end_parity = (T_x + T_y) % 2
    
    # If the parities are the same, the toll is the maximum of dx and dy
    if start_parity == end_parity:
        return max(dx, dy)
    else:
        # If the parities are different, the toll is the maximum of dx and dy plus 1
        return max(dx, dy) + 1

# Read input
S_x, S_y = map(int, input().split())
T_x, T_y = map(int, input().split())

# Compute and print the result
print(min_toll(S_x, S_y, T_x, T_y))