def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    H = int(data[0])
    W = int(data[1])
    grid = data[2:]
    
    # Find the bounding rectangle where cookies were originally placed.
    top = H
    bottom = -1
    left = W
    right = -1
    for i in range(H):
        for j in range(W):
            if grid[i][j] == "#":
                if i < top:
                    top = i
                if i > bottom:
                    bottom = i
                if j < left:
                    left = j
                if j > right:
                    right = j

    # In the original rectangle every cell had a cookie.
    # Snuke removed one cookie so exactly one cell in the bounding rectangle will be '.'
    missing_i, missing_j = -1, -1
    for i in range(top, bottom + 1):
        for j in range(left, right + 1):
            if grid[i][j] == ".":
                missing_i = i + 1  # converting to 1-indexed
                missing_j = j + 1
                break
        if missing_i != -1:
            break

    print(missing_i, missing_j)

if __name__ == '__main__':
    main()