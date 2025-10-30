# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    W = int(data[0])
    H = int(data[1])
    N = int(data[2])
    
    strawberries = []
    index = 3
    for _ in range(N):
        p = int(data[index])
        q = int(data[index + 1])
        strawberries.append((p, q))
        index += 2
    
    A = int(data[index])
    index += 1
    a_lines = list(map(int, data[index:index + A]))
    index += A
    
    B = int(data[index])
    index += 1
    b_lines = list(map(int, data[index:index + B]))
    
    # Add boundaries to the lines
    a_lines = [0] + a_lines + [W]
    b_lines = [0] + b_lines + [H]
    
    # Create a dictionary to count strawberries in each piece
    piece_count = {}
    
    for p, q in strawberries:
        # Find the piece where the strawberry (p, q) belongs
        x_piece = next(i for i in range(len(a_lines) - 1) if a_lines[i] < p < a_lines[i + 1])
        y_piece = next(i for i in range(len(b_lines) - 1) if b_lines[i] < q < b_lines[i + 1])
        
        piece = (x_piece, y_piece)
        if piece in piece_count:
            piece_count[piece] += 1
        else:
            piece_count[piece] = 1
    
    # Find the minimum and maximum number of strawberries in any piece
    min_strawberries = min(piece_count.values(), default=0)
    max_strawberries = max(piece_count.values(), default=0)
    
    print(min_strawberries, max_strawberries)

if __name__ == "__main__":
    main()