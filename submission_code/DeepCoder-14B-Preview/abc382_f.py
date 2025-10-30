def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    H = int(input[ptr]); ptr +=1
    W = int(input[ptr]); ptr +=1
    N = int(input[ptr]); ptr +=1

    bars = []
    for _ in range(N):
        R = int(input[ptr]); ptr +=1
        C = int(input[ptr]); ptr +=1
        L = int(input[ptr]); ptr +=1
        bars.append( (R, C, L) )

    # We need to process bars in the original order (1 to N)
    # and for each, move it down as far as possible, considering the already placed bars.

    occupied = set()
    results = [0] * N  # results[i] will store the final row for the (i+1)-th bar (0-based index)

    for i in range(N):
        R, C, L = bars[i]
        current_row = R
        # Add initial position to occupied
        for c in range(C, C + L):
            occupied.add( (current_row, c) )
        # Try to move down as far as possible
        while True:
            next_row = current_row + 1
            if next_row > H:
                break
            # Check if all cells in next_row, C to C+L-1 are unoccupied
            can_move = True
            for c in range(C, C + L):
                if (next_row, c) in occupied:
                    can_move = False
                    break
            if can_move:
                # Remove current_row cells
                for c in range(C, C + L):
                    occupied.remove( (current_row, c) )
                current_row = next_row
                # Add next_row cells
                for c in range(C, C + L):
                    occupied.add( (current_row, c) )
            else:
                break
        results[i] = current_row

    for res in results:
        print(res)

if __name__ == '__main__':
    main()