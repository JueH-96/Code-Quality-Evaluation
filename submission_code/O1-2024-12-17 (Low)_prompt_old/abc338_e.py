def solve():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    
    chords = []
    idx = 1
    for _ in range(N):
        A, B = int(input_data[idx]), int(input_data[idx+1])
        idx += 2
        if A < B:
            chords.append((A, B))
        else:
            chords.append((B, A))
    
    # Sort chords by their left (smaller) endpoint
    chords.sort(key=lambda x: x[0])
    
    # Fenwick (BIT) for counting how many endpoints have appeared
    # up to a certain position. We have positions 1..(2N).
    size = 2*N
    fenwicks = [0]*(size+1)
    
    def fenwicks_update(pos, val):
        while pos <= size:
            fenwicks[pos] += val
            pos += (pos & -pos)
    
    def fenwicks_query(pos):
        s = 0
        while pos > 0:
            s += fenwicks[pos]
            pos -= (pos & -pos)
        return s
    
    def fenwicks_range_query(left, right):
        if right < left:
            return 0
        return fenwicks_query(right) - fenwicks_query(left-1)
    
    # Process each chord in ascending order of left endpoint.
    # For each chord (L, R), check how many R' are strictly between L and R
    # (i.e. in (L, R)) among previously processed chords.
    # If we find any, that means we have an intersection.
    for L, R in chords:
        # Query how many endpoints are between L+1 and R-1
        if R - 1 >= L + 1:
            if fenwicks_range_query(L+1, R-1) > 0:
                print("Yes")
                return
        # Insert the right endpoint into Fenwick
        fenwicks_update(R, 1)
    
    # If no intersections found
    print("No")

def main():
    solve()

if __name__ == "__main__":
    main()