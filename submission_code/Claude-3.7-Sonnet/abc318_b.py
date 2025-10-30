def calculate_covered_area(rectangles):
    # Create a set to store the covered cells
    covered_cells = set()
    
    # For each rectangle, add all the cells it covers to the set
    for a, b, c, d in rectangles:
        for x in range(a, b):
            for y in range(c, d):
                covered_cells.add((x, y))
    
    # Return the number of unique cells covered
    return len(covered_cells)

# Read the number of rectangles
N = int(input())

# Read the coordinates of each rectangle
rectangles = []
for _ in range(N):
    a, b, c, d = map(int, input().split())
    rectangles.append((a, b, c, d))

# Calculate and print the covered area
print(calculate_covered_area(rectangles))