import sys
from bisect import bisect_right
from collections import defaultdict

def solve():
    # Read W and H (cake dimensions, not directly used for piece identification
    # logic, but part of the input specification)
    W, H = map(int, sys.stdin.readline().split())
    
    # Read N, the total number of strawberries
    N = int(sys.stdin.readline())
    
    # Read and store all strawberry coordinates.
    # It's important to read all strawberries first, as the cut lines (a_coords, b_coords)
    # are provided after the strawberry coordinates in the input format.
    strawberries = []
    for _ in range(N):
        p, q = map(int, sys.stdin.readline().split())
        strawberries.append((p, q))

    # Read A, the number of vertical cuts, and the x-coordinates of these cuts.
    # The problem guarantees a_coords are sorted: 0 < a_1 < ... < a_A < W.
    A = int(sys.stdin.readline())
    a_coords = list(map(int, sys.stdin.readline().split()))

    # Read B, the number of horizontal cuts, and the y-coordinates of these cuts.
    # The problem guarantees b_coords are sorted: 0 < b_1 < ... < b_B < H.
    B = int(sys.stdin.readline())
    b_coords = list(map(int, sys.stdin.readline().split()))

    # Use a defaultdict to count strawberries in each rectangular piece.
    # A piece can be uniquely identified by its (x_idx, y_idx) tuple.
    # x_idx: 0-indexed column number.
    # y_idx: 0-indexed row number.
    # For example, (0,0) would be the piece [0, a_1) x [0, b_1).
    piece_counts = defaultdict(int)

    # Iterate through each strawberry to determine which piece it belongs to
    # and update the count for that piece.
    for p, q in strawberries:
        # For a strawberry at x-coordinate 'p', bisect_right(a_coords, p)
        # returns the index 'idx' such that all elements a_coords[k] for k < idx
        # are less than or equal to 'p', and all elements a_coords[k] for k >= idx
        # are greater than 'p'.
        # This 'idx' effectively represents the 0-indexed column of the piece.
        # E.g., if a_coords = [2, 5]:
        #   - If p = 1 (0 < p < 2), bisect_right returns 0. (Piece in [0, 2) x ...)
        #   - If p = 3 (2 < p < 5), bisect_right returns 1. (Piece in [2, 5) x ...)
        #   - If p = 6 (5 < p < W), bisect_right returns 2. (Piece in [5, W) x ...)
        x_idx = bisect_right(a_coords, p)
        
        # Apply the same logic for the y-coordinate 'q' to find the row index.
        y_idx = bisect_right(b_coords, q)
        
        # Increment the count for the piece identified by (x_idx, y_idx).
        piece_counts[(x_idx, y_idx)] += 1

    # Calculate the minimum and maximum number of strawberries.
    
    # The maximum number of strawberries in any piece is simply the largest value
    # found in our piece_counts dictionary.
    # Since N >= 1 (constraint), piece_counts will not be empty, so max() is safe.
    max_strawberries = max(piece_counts.values())

    # To find the minimum number of strawberries, we must consider pieces that
    # might contain zero strawberries.
    # The total number of distinct rectangular pieces formed is (A+1) * (B+1).
    total_possible_pieces = (A + 1) * (B + 1)
    
    # The number of pieces that actually contain at least one strawberry
    # is the number of unique keys in our piece_counts dictionary.
    num_non_empty_pieces = len(piece_counts)

    # If the number of pieces with strawberries is less than the total possible pieces,
    # it implies that there are some pieces that have 0 strawberries.
    if num_non_empty_pieces < total_possible_pieces:
        min_strawberries = 0
    else:
        # If all (A+1)*(B+1) pieces contain at least one strawberry, then
        # the minimum is the smallest count among those pieces.
        min_strawberries = min(piece_counts.values())

    # Print the minimum and maximum results, space-separated.
    print(min_strawberries, max_strawberries)

# Call the solve function to execute the program logic.
solve()