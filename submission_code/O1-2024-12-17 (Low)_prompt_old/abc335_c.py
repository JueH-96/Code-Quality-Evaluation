def solve():
    import sys
    input_data = sys.stdin.read().strip().split()
    
    # Parse N and Q
    N = int(input_data[0])
    Q = int(input_data[1])
    
    # Prepare arrays to store coordinates of each part.
    # part i (1-indexed) will be stored at index i-1.
    # So X[i-1], Y[i-1] will hold the position of part i.
    X = [0]*(N)
    Y = [0]*(N)
    
    # Initialize positions:
    # part i starts at (i, 0)
    for i in range(N):
        X[i] = i+1
        Y[i] = 0
    
    # offset will mark which index corresponds to the head (part 1).
    offset = 0
    
    # Direction map
    dir_map = {
        'R': (1, 0),
        'L': (-1, 0),
        'U': (0, 1),
        'D': (0, -1)
    }
    
    # We'll process the queries in order.
    idx = 2  # We have used input_data[0..1] for N, Q, so start reading from input_data[2].
    
    out = []
    modN = N  # just to clarify that we mod by N
    
    for _ in range(Q):
        t = input_data[idx]
        idx += 1
        
        if t == '1':
            # Move query
            C = input_data[idx]
            idx += 1
            dx, dy = dir_map[C]
            
            # old head
            old_head_x = X[offset]
            old_head_y = Y[offset]
            
            # shift offset (the chain shifts so that what used to be the head
            # becomes part2, etc., effectively offset moves -1)
            offset = (offset - 1) % modN
            
            # new head position
            X[offset] = old_head_x + dx
            Y[offset] = old_head_y + dy
        
        else:
            # t == '2': position query
            p = int(input_data[idx])
            idx += 1
            # part p is at index (offset + (p-1)) mod N
            actual_index = (offset + (p-1)) % modN
            out.append(f"{X[actual_index]} {Y[actual_index]}")
    
    print("
".join(out))

def main():
    solve()

if __name__ == "__main__":
    main()