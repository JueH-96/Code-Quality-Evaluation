# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    H = int(data[0])
    W = int(data[1])
    N = int(data[2])
    T = data[3]
    S = data[4:]
    
    # Initialize possible positions
    possible_positions = set()
    for i in range(1, H-1):
        for j in range(1, W-1):
            if S[i][j] == '.':
                possible_positions.add((i, j))
    
    # Reverse the moves
    for move in reversed(T):
        new_positions = set()
        for (i, j) in possible_positions:
            if move == 'L' and j + 1 < W and S[i][j + 1] == '.':
                new_positions.add((i, j + 1))
            elif move == 'R' and j - 1 >= 0 and S[i][j - 1] == '.':
                new_positions.add((i, j - 1))
            elif move == 'U' and i + 1 < H and S[i + 1][j] == '.':
                new_positions.add((i + 1, j))
            elif move == 'D' and i - 1 >= 0 and S[i - 1][j] == '.':
                new_positions.add((i - 1, j))
        possible_positions = new_positions
    
    print(len(possible_positions))

if __name__ == "__main__":
    main()