import sys
import collections
import bisect

def solve():
    # W and H define the overall cake dimensions. They are not strictly needed
    # for determining which piece a strawberry falls into, as the cut lines
    # a_lines and b_lines along with bisect_left handle this.
    # The outermost pieces extend to W and H, but their indices are correctly
    # determined by bisect_left (e.g., index A for x-coordinate > a_A).
    _ = sys.stdin.readline() # Read and discard W, H line

    N = int(sys.stdin.readline())
    
    strawberries = []
    for _ in range(N):
        p, q = map(int, sys.stdin.readline().split())
        strawberries.append((p, q))
        
    A_count = int(sys.stdin.readline())
    # Constraints: 1 <= A, B. So A_count >= 1.
    # The problem statement says a_1, ..., a_A are distinct and sorted.
    a_lines = list(map(int, sys.stdin.readline().split()))
        
    B_count = int(sys.stdin.readline())
    # Constraints: B_count >= 1.
    # The problem statement says b_1, ..., b_B are distinct and sorted.
    b_lines = list(map(int, sys.stdin.readline().split()))

    # Using collections.Counter to store counts of strawberries per piece.
    # A piece is identified by a tuple (idx_x, idx_y).
    counts = collections.Counter()
    
    for p, q in strawberries:
        # Find which x-strip the strawberry (p,q) is in.
        # bisect_left returns an index from 0 to A_count.
        # idx_x = 0 means p < a_lines[0]
        # idx_x = k (0 < k < A_count) means a_lines[k-1] < p < a_lines[k] (since p != a_lines[j])
        # idx_x = A_count means p > a_lines[A_count-1]
        idx_x = bisect.bisect_left(a_lines, p)
        
        # Similarly for y-strip.
        idx_y = bisect.bisect_left(b_lines, q)
        
        counts[(idx_x, idx_y)] += 1
        
    # Constraints: N >= 1. So 'counts' will not be empty.
    # Therefore, counts.values() will not be empty.
    
    # Initialize min_val_among_non_empty_pieces to a value larger than any possible count.
    # Max possible strawberries in any single piece is N.
    min_val_among_non_empty_pieces = N + 1 
    # Initialize max_strawberries. Since N >= 1, any piece with strawberries has at least 1.
    max_strawberries = 0 
    
    for count_val in counts.values():
        if count_val < min_val_among_non_empty_pieces:
            min_val_among_non_empty_pieces = count_val
        if count_val > max_strawberries:
            max_strawberries = count_val
            
    # Determine the final minimum number of strawberries on a chosen piece.
    # This could be 0 if some pieces are empty.
    num_pieces_with_strawberries = len(counts)
    
    # Calculate total number of pieces. Python handles large integers automatically.
    total_pieces = (A_count + 1) * (B_count + 1)
    
    min_final_strawberries = 0 # This will be the answer for min if some pieces are empty
    if num_pieces_with_strawberries < total_pieces:
        # There's at least one piece with 0 strawberries.
        min_final_strawberries = 0
    else:
        # All (A_count+1)*(B_count+1) pieces have strawberries.
        # The minimum is min_val_among_non_empty_pieces.
        min_final_strawberries = min_val_among_non_empty_pieces
    
    sys.stdout.write(f"{min_final_strawberries} {max_strawberries}
")

if __name__ == '__main__':
    solve()