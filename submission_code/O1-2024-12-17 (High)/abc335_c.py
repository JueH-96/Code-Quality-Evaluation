def main():
    import sys
    
    input_data = sys.stdin.read().strip().split()
    N, Q = map(int, input_data[:2])
    
    # Prefix sums of the moves (dx, dy). P[t] = v1 + v2 + ... + v_t.
    # We'll store them separately in prefix_x and prefix_y.
    prefix_x = [0] * (Q + 1)
    prefix_y = [0] * (Q + 1)
    
    # Number of moves so far
    T = 0  
    idx = 2  # Pointer into input_data beyond N,Q
    
    # A small helper to map direction letter to (dx, dy)
    def get_move(c):
        if c == 'R': return (1, 0)
        if c == 'L': return (-1, 0)
        if c == 'U': return (0, 1)
        if c == 'D': return (0, -1)
        return (0, 0)  # Shouldn't happen
    
    # We'll collect all answers here and output at the end
    answers = []
    
    for _ in range(Q):
        typ = input_data[idx]
        idx += 1
        
        if typ == '1':
            # Move query
            c = input_data[idx]
            idx += 1
            dx, dy = get_move(c)
            T += 1
            prefix_x[T] = prefix_x[T - 1] + dx
            prefix_y[T] = prefix_y[T - 1] + dy
            
        else:
            # Query for part p
            p = int(input_data[idx])
            idx += 1
            M = p - 1  # We want the position of part p
            
            # If T < N-1, some of the original "horizontal offsets" (1,0) still remain.
            # Otherwise, they've all been popped off.
            if T < N - 1:
                # Case 1: Not all original differences are gone
                if M <= T:
                    # part p = (1,0) + prefix_sums[T - M]
                    x = 1 + prefix_x[T - M]
                    y = prefix_y[T - M]
                else:
                    # If p-1 > T, the part hasn't fully been pulled away from its initial spot
                    # and is still on the x-axis at x = p - T
                    x = p - T
                    y = 0
            else:
                # Case 2: T >= N-1 => all original differences are gone
                # part p = (1,0) + prefix_sums[T - M]
                x = 1 + prefix_x[T - M]
                y = prefix_y[T - M]
            
            answers.append(f"{x} {y}")
    
    print("
".join(answers))

# Do not forget to call main().
if __name__ == "__main__":
    main()