# YOUR CODE HERE
import sys
input = sys.stdin.read

def main():
    data = input().split()
    N = int(data[0])
    Q = int(data[1])
    
    queries = data[2:]
    
    # Initial colors of the cells
    colors = list(range(1, N + 1))
    
    # Dictionary to keep track of the count of each color
    color_count = {i: 1 for i in range(1, N + 1)}
    
    output = []
    i = 0
    while i < len(queries):
        if queries[i] == '1':
            x = int(queries[i + 1]) - 1
            c = int(queries[i + 2])
            old_color = colors[x]
            if old_color != c:
                # BFS to repaint all reachable cells
                queue = [x]
                while queue:
                    current = queue.pop(0)
                    if colors[current] == old_color:
                        colors[current] = c
                        color_count[old_color] -= 1
                        color_count[c] = color_count.get(c, 0) + 1
                        if current > 0 and colors[current - 1] == old_color:
                            queue.append(current - 1)
                        if current < N - 1 and colors[current + 1] == old_color:
                            queue.append(current + 1)
            i += 3
        elif queries[i] == '2':
            c = int(queries[i + 1])
            output.append(str(color_count.get(c, 0)))
            i += 2
    
    print("
".join(output))

if __name__ == "__main__":
    main()