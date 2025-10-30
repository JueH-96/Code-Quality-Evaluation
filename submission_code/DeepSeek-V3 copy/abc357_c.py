def generate_carpet(n):
    if n == 0:
        return ['#']
    
    # Recursively get the smaller carpet
    smaller_carpet = generate_carpet(n - 1)
    size = len(smaller_carpet)
    
    # Initialize the new carpet
    new_carpet = []
    
    # Top three rows
    for row in smaller_carpet:
        new_carpet.append(row * 3)
    
    # Middle three rows
    for row in smaller_carpet:
        new_row = row + '.' * size + row
        new_carpet.append(new_row)
    
    # Bottom three rows
    for row in smaller_carpet:
        new_carpet.append(row * 3)
    
    return new_carpet

# Read input
N = int(input())

# Generate the carpet
carpet = generate_carpet(N)

# Print the carpet
for row in carpet:
    print(row)