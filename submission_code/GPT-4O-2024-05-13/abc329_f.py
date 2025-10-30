# YOUR CODE HERE
import sys
input = sys.stdin.read

def main():
    data = input().split()
    N = int(data[0])
    Q = int(data[1])
    
    colors = list(map(int, data[2:N+2]))
    queries = data[N+2:]
    
    # Initialize boxes with their respective colors
    boxes = [set() for _ in range(N)]
    for i in range(N):
        boxes[i].add(colors[i])
    
    result = []
    for i in range(Q):
        a = int(queries[2*i]) - 1
        b = int(queries[2*i + 1]) - 1
        
        # Move all balls from box a to box b
        boxes[b].update(boxes[a])
        boxes[a].clear()
        
        # Print the number of different colors in box b
        result.append(str(len(boxes[b])))
    
    print("
".join(result))

if __name__ == "__main__":
    main()