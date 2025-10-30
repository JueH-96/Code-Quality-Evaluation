import sys
from collections import defaultdict, deque

def main():
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    Q = int(data[idx+1])
    idx += 2
    
    # Initialize color of each cell
    color = [i+1 for i in range(N)]
    
    # To count the number of cells for each color
    color_count = defaultdict(int)
    for c in color:
        color_count[c] += 1
    
    for _ in range(Q):
        query_type = int(data[idx])
        if query_type == 1:
            x = int(data[idx+1]) - 1
            c = int(data[idx+2])
            idx += 3
            original_color = color[x]
            if original_color == c:
                continue
            # BFS to find all connected cells with the same color
            queue = deque()
            queue.append(x)
            visited = set()
            visited.add(x)
            while queue:
                current = queue.popleft()
                color[current] = c
                color_count[original_color] -= 1
                color_count[c] += 1
                # Check left neighbor
                if current > 0 and color[current-1] == original_color and (current-1) not in visited:
                    visited.add(current-1)
                    queue.append(current-1)
                # Check right neighbor
                if current < N-1 and color[current+1] == original_color and (current+1) not in visited:
                    visited.add(current+1)
                    queue.append(current+1)
        else:
            c = int(data[idx+1])
            idx += 2
            print(color_count.get(c, 0))

if __name__ == "__main__":
    main()