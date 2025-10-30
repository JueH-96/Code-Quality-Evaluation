def is_black(x, y):
    # Check if point (x, y) is in a black region
    # First handle points exactly on lines
    if x == int(x) or y == int(y) or x + y == int(x + y):
        return False
    
    # Count number of lines crossed from (0.5, 0.5) to (x, y)
    # We know (0.5, 0.5) is black
    crossings = 0
    
    # Vertical lines (x = n)
    min_x = min(0.5, x)
    max_x = max(0.5, x)
    crossings += int(max_x) - int(min_x)
    
    # Horizontal lines (y = n where n is even)
    min_y = min(0.5, y)
    max_y = max(0.5, y)
    crossings += len([i for i in range(int(min_y), int(max_y) + 1) if i % 2 == 0])
    
    # Diagonal lines (x + y = n where n is even)
    min_sum = min(0.5 + 0.5, x + y)
    max_sum = max(0.5 + 0.5, x + y)
    crossings += len([i for i in range(int(min_sum), int(max_sum) + 1) if i % 2 == 0])
    
    # If odd number of crossings, color changes from black
    return crossings % 2 == 0

def solve(a, b, c, d):
    # Split rectangle into unit squares and calculate black area in each
    total = 0
    for x in range(a, c):
        for y in range(b, d):
            # Check corners of unit square
            corners = [
                is_black(x + 0.5, y + 0.5),
                is_black(x + 0.5, y + 1.5),
                is_black(x + 1.5, y + 0.5),
                is_black(x + 1.5, y + 1.5)
            ]
            # Area is proportional to number of black corners
            total += sum(corners) / 4

    return int(total * 2)

A, B, C, D = map(int, input().split())
print(solve(A, B, C, D))