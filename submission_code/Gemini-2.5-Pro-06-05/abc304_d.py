import sys
import collections
import bisect

# Read and discard the W, H line, as these large values are not directly
# needed for the logic. The number of pieces is determined by the number of cuts.
sys.stdin.readline()

# Read the number of strawberries.
try:
    N = int(sys.stdin.readline())
except (ValueError, IndexError):
    # This handles potential empty input during local testing,
    # though contest environments guarantee valid input.
    N = 0

if N == 0:
    # If there are no strawberries, read the rest of the input to clear the buffer
    # and print 0 0 as both min and max are 0.
    # This case is not expected based on constraints (N>=1), but makes the code robust.
    try:
        A = int(sys.stdin.readline())
        if A > 0: sys.stdin.readline()
        B = int(sys.stdin.readline())
        if B > 0: sys.stdin.readline()
    except (ValueError, IndexError):
        pass
    print(0, 0)
else:
    # Read the coordinates of the N strawberries.
    strawberries = [tuple(map(int, sys.stdin.readline().split())) for _ in range(N)]

    # Read the vertical cuts information.
    A = int(sys.stdin.readline())
    a_cuts = list(map(int, sys.stdin.readline().split()))

    # Read the horizontal cuts information.
    B = int(sys.stdin.readline())
    b_cuts = list(map(int, sys.stdin.readline().split()))

    # Use a Counter to count strawberries in each rectangular piece.
    # The key for the counter will be a tuple (ix, iy) representing the
    # grid cell of the piece.
    piece_counts = collections.Counter()

    # For each strawberry, find which piece it belongs to and update the count.
    for p, q in strawberries:
        # `bisect_left` efficiently finds the index of the region.
        # For a point `p` and sorted cuts `a_cuts`, `bisect_left(a_cuts, p)`
        # gives the number of cuts to the left of `p`. This number is the
        # zero-based index of the vertical strip the point `p` is in.
        ix = bisect.bisect_left(a_cuts, p)
        iy = bisect.bisect_left(b_cuts, q)
        
        # Increment the count for the piece at grid (ix, iy).
        piece_counts[(ix, iy)] += 1

    # Calculate the maximum number of strawberries on any single piece.
    # Since the constraints state N >= 1, `piece_counts` will not be empty,
    # so calling max() on its values is safe.
    max_val = max(piece_counts.values())

    # The minimum number of strawberries is 0, unless every single piece
    # contains at least one strawberry.
    min_val = 0
    total_pieces = (A + 1) * (B + 1)

    # If the number of pieces with strawberries equals the total number of pieces,
    # then all pieces are non-empty. In this case, the minimum is the smallest
    # count found.
    if len(piece_counts) == total_pieces:
        min_val = min(piece_counts.values())

    # Print the minimum and maximum values separated by a space.
    print(min_val, max_val)