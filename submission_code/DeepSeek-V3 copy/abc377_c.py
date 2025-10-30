# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    
    occupied = set()
    capture = set()
    
    for i in range(M):
        a = int(data[2 + 2*i])
        b = int(data[3 + 2*i])
        occupied.add((a, b))
        
        # Generate all possible capture positions
        positions = [
            (a+2, b+1),
            (a+1, b+2),
            (a-1, b+2),
            (a-2, b+1),
            (a-2, b-1),
            (a-1, b-2),
            (a+1, b-2),
            (a+2, b-1)
        ]
        
        for pos in positions:
            x, y = pos
            if 1 <= x <= N and 1 <= y <= N:
                capture.add((x, y))
    
    # Total squares is N*N
    total = N * N
    
    # Subtract the occupied squares
    total -= M
    
    # Subtract the squares that are in capture set but not occupied
    # Because those are the squares that are empty but can be captured
    # So, we need to subtract the number of squares in capture that are not in occupied
    # But since occupied squares are already subtracted, we need to subtract the squares in capture that are not in occupied
    # So, the total is total - (number of squares in capture that are not in occupied)
    
    # To find the number of squares in capture that are not in occupied
    # We can do len(capture - occupied)
    
    # So, total = total - len(capture - occupied)
    
    # But since occupied is a set of occupied squares, and capture is a set of squares that can be captured
    # The squares that are in capture and not in occupied are the squares that are empty but can be captured
    
    # So, the total number of safe squares is total - len(capture - occupied)
    
    # So, we need to compute len(capture - occupied)
    
    # But since capture and occupied are sets, we can do capture - occupied
    
    # So, the number of squares that are in capture but not in occupied is len(capture - occupied)
    
    # So, the total number of safe squares is total - len(capture - occupied)
    
    # So, we need to compute len(capture - occupied)
    
    # So, we can do:
    safe = total - len(capture - occupied)
    
    print(safe)

if __name__ == "__main__":
    main()