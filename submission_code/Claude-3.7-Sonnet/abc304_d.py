import bisect

def solve():
    W, H = map(int, input().split())
    N = int(input())
    
    strawberries = []
    for _ in range(N):
        p, q = map(int, input().split())
        strawberries.append((p, q))
    
    A = int(input())
    a_cuts = list(map(int, input().split()))
    
    B = int(input())
    b_cuts = list(map(int, input().split()))
    
    # Define the boundaries of each piece
    x_boundaries = [0] + a_cuts + [W]
    y_boundaries = [0] + b_cuts + [H]
    
    # Count strawberries in each piece
    counts = {}
    
    for p, q in strawberries:
        # Find which piece the strawberry belongs to using binary search
        i = bisect.bisect_right(x_boundaries, p) - 1
        j = bisect.bisect_right(y_boundaries, q) - 1
        
        key = (i, j)
        counts[key] = counts.get(key, 0) + 1
    
    # Check if all pieces have at least one strawberry
    total_pieces = (A+1) * (B+1)
    pieces_with_strawberries = len(counts)
    
    if pieces_with_strawberries < total_pieces:
        # There's at least one piece without any strawberry
        min_count = 0
    else:
        # All pieces have at least one strawberry
        min_count = min(counts.values())
    
    max_count = max(counts.values()) if counts else 0
    
    return min_count, max_count

min_strawberries, max_strawberries = solve()
print(min_strawberries, max_strawberries)