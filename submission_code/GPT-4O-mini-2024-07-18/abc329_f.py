def main():
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    
    # Read N and Q
    N, Q = map(int, data[0].split())
    
    # Read colors of the balls in each box
    colors = list(map(int, data[1].split()))
    
    # Initialize boxes with their respective colors
    boxes = [set() for _ in range(N)]
    for i in range(N):
        boxes[i].add(colors[i])
    
    results = []
    
    # Process each query
    for i in range(2, 2 + Q):
        a, b = map(int, data[i].split())
        a -= 1  # Convert to 0-based index
        b -= 1  # Convert to 0-based index
        
        # Move all balls from box a to box b
        if boxes[a]:
            boxes[b].update(boxes[a])
            boxes[a].clear()  # Empty box a after moving
        
        # Count the number of different colors in box b
        results.append(len(boxes[b]))
    
    # Print all results
    sys.stdout.write('
'.join(map(str, results)) + '
')

if __name__ == "__main__":
    main()