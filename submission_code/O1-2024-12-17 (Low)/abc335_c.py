def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    # Parse N, Q
    N, Q = map(int, input_data[:2])
    
    # We will store the (x, y) position of the "head" for "times" from - (N-1) up to at most Q.
    # Indexing trick:
    #   Let SHIFT = N - 1
    #   We store pos[t + SHIFT] = (x, y)  where t can range from -(N-1) up to Q.
    # This array must be large enough: size = N - 1 + Q + 1 (plus a small safety margin).
    
    SHIFT = N - 1
    size = N + Q + 10  # A bit of safety margin
    
    # We'll store coordinates in two separate lists for (x, y) to be more memory-friendly.
    posx = [0]*(size)
    posy = [0]*(size)
    
    # Initialize the positions at "times" t = -(N-1), -(N-2), ..., -1, 0.
    # At time 0, the head is at (1,0).
    # At time -1, "would-be" the head is at (2,0).
    # Generally, pos[-(i-1)] = (i, 0) for i in [1..N].
    # So index = SHIFT + (-(i-1)) = SHIFT - (i-1) = SHIFT - i + 1.
    # i from 1..N
    for i in range(1, N+1):
        idx = SHIFT - (i - 1)
        posx[idx] = i
        posy[idx] = 0
    
    # Dictionary to map move characters to (dx, dy).
    moves = {'R': (1, 0), 'L': (-1, 0), 'U': (0, 1), 'D': (0, -1)}
    
    # t tracks how many moves (type-1 queries) have happened so far.
    t = 0
    
    # Process each query
    # The queries start at input_data[2], each query is either "1 C" or "2 p".
    # We'll read them in a loop.
    idx_data = 2  # index in input_data
    output = []
    for _ in range(Q):
        query_type = input_data[idx_data]
        idx_data += 1
        
        if query_type == '1':
            # Move query: "1 C"
            C = input_data[idx_data]
            idx_data += 1
            dx, dy = moves[C]
            t += 1
            # pos[t + SHIFT] = pos[t-1 + SHIFT] + (dx, dy)
            px = posx[SHIFT + t - 1] + dx
            py = posy[SHIFT + t - 1] + dy
            posx[SHIFT + t] = px
            posy[SHIFT + t] = py
        
        else:
            # query_type == '2'
            # "2 p": find coordinates of part p
            p = int(input_data[idx_data])
            idx_data += 1
            # The position of part p after t moves is the position that the head had at time (t - (p-1)).
            # So array index = SHIFT + (t - (p-1)) = SHIFT + t + 1 - p
            arr_index = SHIFT + t + 1 - p
            px = posx[arr_index]
            py = posy[arr_index]
            output.append(f"{px} {py}")
    
    # Print all answers
    print("
".join(output))

# Do not forget to call main()
if __name__ == "__main__":
    main()