# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    index = 0
    
    W = int(data[index])
    H = int(data[index + 1])
    index += 2
    
    N = int(data[index])
    index += 1
    
    strawberries = []
    for _ in range(N):
        p = int(data[index])
        q = int(data[index + 1])
        strawberries.append((p, q))
        index += 2
    
    A = int(data[index])
    index += 1
    
    a_cuts = []
    for _ in range(A):
        a_cuts.append(int(data[index]))
        index += 1
    
    B = int(data[index])
    index += 1
    
    b_cuts = []
    for _ in range(B):
        b_cuts.append(int(data[index]))
        index += 1
    
    # Add boundaries to the cut lists
    a_cuts = [0] + a_cuts + [W]
    b_cuts = [0] + b_cuts + [H]
    
    # Create a dictionary to count strawberries in each piece
    from collections import defaultdict
    piece_count = defaultdict(int)
    
    # Determine which piece each strawberry belongs to
    for p, q in strawberries:
        # Find the x-segment
        x_segment = next(i for i in range(len(a_cuts) - 1) if a_cuts[i] < p < a_cuts[i + 1])
        # Find the y-segment
        y_segment = next(i for i in range(len(b_cuts) - 1) if b_cuts[i] < q < b_cuts[i + 1])
        
        # Count this strawberry in the corresponding piece
        piece_count[(x_segment, y_segment)] += 1
    
    # Find the minimum and maximum number of strawberries in any piece
    if piece_count:
        min_strawberries = min(piece_count.values())
        max_strawberries = max(piece_count.values())
    else:
        min_strawberries = 0
        max_strawberries = 0
    
    print(min_strawberries, max_strawberries)

if __name__ == "__main__":
    main()