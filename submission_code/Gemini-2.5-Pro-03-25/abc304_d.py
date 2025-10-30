# YOUR CODE HERE
import sys
from bisect import bisect_right
# from collections import defaultdict # Alternative dictionary initialization, not strictly necessary

def solve():
    """
    Solves the problem of finding the minimum and maximum number of strawberries
    in any piece after cutting a rectangular cake. Reads input from stdin,
    computes the result, and prints to stdout.
    """

    # Read cake dimensions W, H. These values define the boundaries 0 <= x <= W and 0 <= y <= H.
    # However, W and H are not explicitly needed in the calculations because the cuts a_i and b_i
    # implicitly define the pieces relative to the boundaries 0 and W/H.
    # We still need to read this line to consume the input correctly.
    # W, H = map(int, sys.stdin.readline().split()) # uncomment if W, H are needed later
    sys.stdin.readline() # Read and discard the W H line

    # Read the total number of strawberries N
    N = int(sys.stdin.readline())
    
    # Read the coordinates of N strawberries.
    # Store them in a list. Each element can be a list [p_i, q_i] or tuple (p_i, q_i).
    strawberries = []
    # Optimization: Only read coordinates if there are strawberries (N > 0).
    if N > 0: 
        for _ in range(N):
            strawberries.append(list(map(int, sys.stdin.readline().split())))

    # Read the number of vertical cuts A
    A = int(sys.stdin.readline())
    # Read the A coordinates of vertical cuts (lines x = a_i).
    # The constraints guarantee A >= 1, so we don't need to check A > 0 strictly.
    # But handling edge cases like A=0 (if constraints allowed) would look like this:
    if A > 0:
      a = list(map(int, sys.stdin.readline().split()))
    else:
      # This branch is technically unreachable due to constraints A >= 1.
      a = [] 

    # Read the number of horizontal cuts B
    B = int(sys.stdin.readline())
    # Read the B coordinates of horizontal cuts (lines y = b_i).
    # Constraints guarantee B >= 1.
    if B > 0:
      b = list(map(int, sys.stdin.readline().split()))
    else:
      # This branch is technically unreachable due to constraints B >= 1.
      b = []

    # Dictionary to store counts of strawberries per piece.
    # Key: tuple (j, k) where j is the index of the vertical strip (0 to A)
    # and k is the index of the horizontal strip (0 to B).
    # Value: Integer count of strawberries in the piece defined by (j, k).
    counts = {}
    # An alternative using collections.defaultdict(int) could simplify the increment step slightly.
    # counts = defaultdict(int) 

    # Iterate through each strawberry to determine which piece it belongs to
    # and update the count for that piece.
    # This loop does nothing if N=0 because the `strawberries` list will be empty.
    for p, q in strawberries:
        # Find the index j of the vertical strip that contains the strawberry's x-coordinate p.
        # The vertical strips are defined by the lines x=0, x=a_1, ..., x=a_A, x=W.
        # The j-th strip (0-indexed) is the region x_j < x < x_{j+1}, where x_0=0, x_i=a_i for 1<=i<=A, x_{A+1}=W.
        # `bisect_right(a, p)` finds the insertion point for p in the sorted list `a`.
        # This insertion point `j` corresponds exactly to the index of the vertical strip.
        # E.g., if p < a_1, `bisect_right` returns 0. If a_j < p < a_{j+1}, returns j. If p > a_A, returns A.
        j = bisect_right(a, p)
        
        # Similarly, find the index k of the horizontal strip for the y-coordinate q.
        # The horizontal strips are defined by y=0, y=b_1, ..., y=b_B, y=H.
        # The k-th strip (0-indexed) is y_k < y < y_{k+1}.
        # `bisect_right(b, q)` gives the correct index k.
        k = bisect_right(b, q)
        
        # The piece is uniquely identified by the pair of indices (j, k).
        piece_key = (j, k)
        
        # Increment the strawberry count for this piece.
        # Using `dict.get(key, default)` avoids KeyError if the key is new.
        counts[piece_key] = counts.get(piece_key, 0) + 1
        # If using defaultdict: counts[piece_key] += 1

    # Initialize minimum and maximum strawberry counts found in any piece.
    min_count = 0
    max_count = 0

    # Calculate the minimum and maximum counts based on the processed strawberry data.
    # This block only executes if there were any strawberries to count (N > 0).
    if N > 0:
        # The maximum count is the largest value found in the `counts` dictionary.
        # Since N > 0, `counts` is guaranteed non-empty, so max() works.
        max_count = max(counts.values())
        
        # Determine the number of distinct pieces that contain at least one strawberry.
        # This is simply the number of entries in the `counts` dictionary.
        num_non_empty_pieces = len(counts)
        
        # Calculate the total number of pieces the cake is divided into.
        # This is (A+1) vertical strips * (B+1) horizontal strips.
        # Python's arbitrary precision integers handle potentially large results.
        total_pieces = (A + 1) * (B + 1)

        # Determine the minimum count. There are two possibilities:
        # 1. If some pieces are empty (contain 0 strawberries).
        # 2. If all pieces contain at least one strawberry.
        if num_non_empty_pieces < total_pieces:
            # If the number of pieces with strawberries is less than the total number of pieces,
            # then there must be at least one piece with 0 strawberries.
            # In this case, the minimum possible count is 0.
            min_count = 0
        else:
            # If num_non_empty_pieces == total_pieces, it means every single piece
            # contains at least one strawberry.
            # The minimum count is then the minimum value present in the `counts` dictionary.
            min_count = min(counts.values())
    
    # If N == 0, the `counts` dictionary is empty. The `if N > 0:` block is skipped.
    # `min_count` remains 0 and `max_count` remains 0. This correctly handles the case
    # where there are no strawberries, resulting in 0 strawberries in every piece.

    # Print the final result: the minimum count and maximum count, separated by a space.
    print(f"{min_count} {max_count}")

# Execute the main function to solve the problem.
solve()