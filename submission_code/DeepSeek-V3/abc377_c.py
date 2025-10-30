# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    
    pieces = []
    index = 2
    for _ in range(M):
        a = int(data[index])
        b = int(data[index+1])
        pieces.append((a, b))
        index += 2
    
    # Initialize a set to store all squares that are under attack
    under_attack = set()
    
    # Define the 8 possible moves
    moves = [ (2, 1), (1, 2), (-1, 2), (-2, 1),
              (-2, -1), (-1, -2), (1, -2), (2, -1) ]
    
    for a, b in pieces:
        for dx, dy in moves:
            x = a + dx
            y = b + dy
            if 1 <= x <= N and 1 <= y <= N:
                under_attack.add((x, y))
    
    # Total squares is N * N
    total_squares = N * N
    
    # Number of squares under attack
    attacked_count = len(under_attack)
    
    # Number of squares occupied by pieces
    occupied_count = M
    
    # The number of safe squares is total_squares - (attacked_count + occupied_count)
    # But since some attacked squares may be occupied, we need to subtract only the unique squares
    # So, the total unsafe squares is the union of under_attack and pieces positions
    # So, total unsafe = len(under_attack | set(pieces))
    
    # Create a set of pieces positions
    pieces_set = set(pieces)
    
    # Union of under_attack and pieces_set
    unsafe = under_attack | pieces_set
    
    # Number of unsafe squares
    unsafe_count = len(unsafe)
    
    # Safe squares
    safe_squares = total_squares - unsafe_count
    
    print(safe_squares)

if __name__ == "__main__":
    main()