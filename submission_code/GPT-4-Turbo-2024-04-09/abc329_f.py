import sys
input = sys.stdin.read
from collections import defaultdict

def main():
    data = input().split()
    index = 0
    
    N = int(data[index])
    index += 1
    Q = int(data[index])
    index += 1
    
    colors = list(map(int, data[index:index+N]))
    index += N
    
    queries = []
    for _ in range(Q):
        a = int(data[index])
        index += 1
        b = int(data[index])
        index += 1
        queries.append((a, b))
    
    # Each box will be represented by a dictionary of color counts
    boxes = [defaultdict(int) for _ in range(N + 1)]
    
    # Initialize the boxes with their initial colors
    for i in range(1, N + 1):
        boxes[i][colors[i - 1]] = 1
    
    results = []
    
    for a, b in queries:
        if a == b:
            results.append(len(boxes[b]))
            continue
        
        # Move all balls from box a to box b
        for color, count in list(boxes[a].items()):
            if count > 0:
                boxes[b][color] += count
                del boxes[a][color]
        
        # Output the number of different colors in box b
        results.append(len(boxes[b]))
    
    # Print all results
    for result in results:
        print(result)

if __name__ == "__main__":
    main()