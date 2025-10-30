# YOUR CODE HERE
import sys

def main():
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    Q = int(data[idx+1])
    idx += 2
    C = list(map(int, data[idx:idx+N]))
    idx += N
    queries = []
    for _ in range(Q):
        a = int(data[idx])
        b = int(data[idx+1])
        queries.append((a, b))
        idx += 2
    
    # Initialize boxes
    boxes = [set() for _ in range(N+1)]
    for i in range(1, N+1):
        boxes[i].add(C[i-1])
    
    for a, b in queries:
        # Move all balls from a to b
        boxes[b].update(boxes[a])
        boxes[a].clear()
        # Print the number of different colors in b
        print(len(boxes[b]))

if __name__ == "__main__":
    main()