def generate_carpet(n):
    if n == 0:
        return ["#"]
    
    # Get previous level carpet
    prev = generate_carpet(n-1)
    size = len(prev)
    
    # Create new carpet of size 3^n x 3^n
    new_size = size * 3
    carpet = []
    
    # For each row in new carpet
    for i in range(new_size):
        row = ""
        # For each column in new carpet
        for j in range(new_size):
            # If in middle block (both row and col), add white cell
            if (size <= i < size*2) and (size <= j < size*2):
                row += "."
            # Otherwise copy from previous level carpet
            else:
                # Map current position to position in previous carpet
                prev_i = i % size
                prev_j = j % size
                row += prev[prev_i][prev_j]
        carpet.append(row)
    
    return carpet

# Read input
N = int(input())

# Generate and print carpet
result = generate_carpet(N)
for row in result:
    print(row)