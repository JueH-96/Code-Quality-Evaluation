# YOUR CODE HERE
import sys

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    H, W = map(int, data[:2])
    grid = data[2:]
    min_i, max_i = H, 1
    min_j, max_j = W, 1
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '#':
                if i+1 < min_i:
                    min_i = i+1
                if i+1 > max_i:
                    max_i = i+1
                if j+1 < min_j:
                    min_j = j+1
                if j+1 > max_j:
                    max_j = j+1
    missing = None
    count_missing = 0
    for i in range(min_i-1, max_i):
        for j in range(min_j-1, max_j):
            if grid[i][j] == '.':
                missing = (i+1, j+1)
                count_missing +=1
    # It is guaranteed that there is exactly one missing
    print(missing[0], missing[1])

if __name__ == "__main__":
    main()