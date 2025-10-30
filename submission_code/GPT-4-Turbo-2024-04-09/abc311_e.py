def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    H = int(data[0])
    W = int(data[1])
    N = int(data[2])
    
    holed = set()
    index = 3
    for _ in range(N):
        a = int(data[index]) - 1
        b = int(data[index + 1]) - 1
        holed.add((a, b))
        index += 2
    
    # To count holeless squares
    holeless_count = 0
    
    # We need to check each possible top-left corner (i, j)
    # and for each (i, j), check for all possible sizes n where (i+n-1, j+n-1) is within bounds
    for i in range(H):
        for j in range(W):
            # Check maximum possible n for this (i, j)
            max_n = min(H - i, W - j)
            for n in range(1, max_n + 1):
                # Check if all cells in the square (i, j) to (i+n-1, j+n-1) are holeless
                holeless = True
                for k in range(n):
                    for l in range(n):
                        if (i + k, j + l) in holed:
                            holeless = False
                            break
                    if not holeless:
                        break
                if holeless:
                    holeless_count += 1
                else:
                    # If we find a holed square in this size, larger squares starting from the same (i, j) will also be holed
                    break
    
    print(holeless_count)

if __name__ == "__main__":
    main()