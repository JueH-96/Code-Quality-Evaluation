# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    Q = int(data[1])
    
    # Initial colors in each box
    colors = list(map(int, data[2:N+2]))
    
    # Queries
    queries = []
    index = N + 2
    for _ in range(Q):
        a = int(data[index]) - 1
        b = int(data[index + 1]) - 1
        queries.append((a, b))
        index += 2
    
    # Initialize the boxes with sets of colors
    boxes = [set() for _ in range(N)]
    for i in range(N):
        boxes[i].add(colors[i])
    
    # Process each query
    results = []
    for a, b in queries:
        # Move all balls from box a to box b
        boxes[b].update(boxes[a])
        boxes[a].clear()
        
        # Count the number of different colors in box b
        results.append(len(boxes[b]))
    
    # Print results for each query
    for result in results:
        print(result)

if __name__ == "__main__":
    main()