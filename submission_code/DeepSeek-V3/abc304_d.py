# YOUR CODE HERE
import sys
import bisect

def main():
    W, H = map(int, sys.stdin.readline().split())
    N = int(sys.stdin.readline())
    strawberries = []
    for _ in range(N):
        p, q = map(int, sys.stdin.readline().split())
        strawberries.append((p, q))
    A = int(sys.stdin.readline())
    a_list = list(map(int, sys.stdin.readline().split()))
    B = int(sys.stdin.readline())
    b_list = list(map(int, sys.stdin.readline().split()))
    
    # Sort the strawberries based on x and y coordinates
    strawberries_sorted_x = sorted(strawberries, key=lambda x: x[0])
    strawberries_sorted_y = sorted(strawberries, key=lambda x: x[1])
    
    # Precompute the x and y cuts
    x_cuts = a_list
    y_cuts = b_list
    
    # Create a dictionary to count strawberries in each piece
    piece_count = {}
    
    for p, q in strawberries:
        # Find the x segment
        x_idx = bisect.bisect_left(x_cuts, p)
        # Find the y segment
        y_idx = bisect.bisect_left(y_cuts, q)
        # The piece is (x_idx, y_idx)
        if (x_idx, y_idx) not in piece_count:
            piece_count[(x_idx, y_idx)] = 0
        piece_count[(x_idx, y_idx)] += 1
    
    # Now, find the minimum and maximum counts
    min_count = float('inf')
    max_count = 0
    for key in piece_count:
        if piece_count[key] < min_count:
            min_count = piece_count[key]
        if piece_count[key] > max_count:
            max_count = piece_count[key]
    
    # Also, consider pieces with zero strawberries
    total_pieces = (A + 1) * (B + 1)
    pieces_with_strawberries = len(piece_count)
    if pieces_with_strawberries < total_pieces:
        min_count = 0
    
    print(min_count, max_count)

if __name__ == "__main__":
    main()