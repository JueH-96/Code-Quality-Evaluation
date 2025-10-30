def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    # We have 2N distinct points, each belongs to exactly one chord.
    # pos[x] will store which chord index has an endpoint at point x.
    pos = [-1] * (2*N + 1)
    
    # Read chord endpoints
    idx = 1
    for chord_id in range(1, N+1):
        A = int(input_data[idx]); B = int(input_data[idx+1])
        idx += 2
        pos[A] = chord_id
        pos[B] = chord_id

    # Fenwick (Binary Indexed) Tree for counting "open" chord starts.
    fenwicks = [0] * (2*N + 1)
    
    def fenwicks_add(i, v):
        while i <= 2*N:
            fenwicks[i] += v
            i += i & -i

    def fenwicks_sum(i):
        s = 0
        while i > 0:
            s += fenwicks[i]
            i -= i & -i
        return s

    def fenwicks_range_sum(l, r):
        if l > r:
            return 0
        return fenwicks_sum(r) - fenwicks_sum(l-1)
    
    # first_endpoint[ch] will store the first endpoint index of chord ch
    # or -1 if we haven't encountered it yet.
    first_endpoint = [-1] * (N+1)
    
    # Scan from 1 to 2N in ascending order
    for point in range(1, 2*N + 1):
        ch = pos[point]
        if first_endpoint[ch] == -1:
            # First time seeing chord ch -> open it
            first_endpoint[ch] = point
            fenwicks_add(point, 1)
        else:
            # Second time seeing chord ch -> close it
            start = first_endpoint[ch]
            # Check how many chord openings are strictly between start and point
            s = fenwicks_range_sum(start+1, point-1)
            if s > 0:
                print("Yes")
                return
            # Remove chord ch's opening from the Fenwick tree
            fenwicks_add(start, -1)
    
    # If we finish without finding any intersection
    print("No")

# Do not forget to call main()
main()