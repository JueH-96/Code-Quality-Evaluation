# YOUR CODE HERE
import sys
input = sys.stdin.read

def main():
    data = input().split()
    H = int(data[0])
    W = int(data[1])
    M = int(data[2])
    
    operations = []
    index = 3
    for _ in range(M):
        T = int(data[index])
        A = int(data[index + 1])
        X = int(data[index + 2])
        operations.append((T, A, X))
        index += 3
    
    row_colors = [-1] * H
    col_colors = [-1] * W
    
    for T, A, X in operations:
        if T == 1:
            row_colors[A - 1] = X
        elif T == 2:
            col_colors[A - 1] = X
    
    color_count = {}
    
    for r in range(H):
        for c in range(W):
            if col_colors[c] != -1:
                color = col_colors[c]
            else:
                color = row_colors[r] if row_colors[r] != -1 else 0
            if color in color_count:
                color_count[color] += 1
            else:
                color_count[color] = 1
    
    distinct_colors = sorted(color_count.items())
    
    print(len(distinct_colors))
    for color, count in distinct_colors:
        print(color, count)

if __name__ == "__main__":
    main()