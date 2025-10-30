def main():
    import sys
    
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    # Dictionary to store the positions of each color
    positions = {}
    for i, color in enumerate(A):
        if color not in positions:
            positions[color] = []
        positions[color].append(i)
    
    count = 0
    # For each color, check if the difference in positions is exactly 2
    for color in range(1, N+1):
        p1, p2 = positions[color]
        if abs(p2 - p1) == 2:
            count += 1
    
    print(count)

# Do not remove or rename the function call
main()