import sys
from collections import defaultdict

def main():
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    Q = int(data[idx+1])
    idx += 2
    
    # Initialize color for each cell
    color = [i+1 for i in range(N)]
    
    # To manage the count of each color
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
            # Perform BFS to find all connected cells with the same color
            stack = [x]
            while stack:
                current = stack.pop()
                if color[current] != original_color:
                    continue
                color[current] = c
                color_count[original_color] -= 1
                color_count[c] += 1
                # Check left neighbor
                if current > 0 and color[current-1] == original_color:
                    stack.append(current-1)
                # Check right neighbor
                if current < N-1 and color[current+1] == original_color:
                    stack.append(current+1)
        else:
            c = int(data[idx+1])
            idx += 2
            print(color_count[c])

if __name__ == "__main__":
    main()