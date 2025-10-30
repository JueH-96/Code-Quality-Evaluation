# YOUR CODE HERE
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
    
    # Initialize prefix sum array
    prefix = [[0] * (W + 1) for _ in range(H + 1)]
    
    # Fill the prefix sum array
    for i in range(H):
        for j in range(W):
            prefix[i + 1][j + 1] = prefix[i + 1][j] + prefix[i][j + 1] - prefix[i][j]
            if (i, j) in holed:
                prefix[i + 1][j + 1] += 1
    
    total_holeless_squares = 0
    
    # Check for each possible size of the square
    for n in range(1, min(H, W) + 1):
        for i in range(H - n + 1):
            for j in range(W - n + 1):
                # Calculate the number of holed squares in the subgrid
                holed_count = (prefix[i + n][j + n]
                               - prefix[i][j + n]
                               - prefix[i + n][j]
                               + prefix[i][j])
                if holed_count == 0:
                    total_holeless_squares += 1
    
    print(total_holeless_squares)

if __name__ == "__main__":
    main()