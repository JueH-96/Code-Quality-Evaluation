def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    T = int(input[ptr])
    ptr += 1
    for _ in range(T):
        N = int(input[ptr])
        ptr += 1
        ABC = []
        for _ in range(N):
            A = int(input[ptr])
            B = int(input[ptr+1])
            C = int(input[ptr+2])
            ptr += 3
            ABC.append((A, B, C))
        
        if N == 0:
            print(0)
            continue
        
        # Initialize y = 1
        y = 1
        # Compute initial x_max
        x_max = min((C - B -1) // A for A, B, C in ABC)
        total = 0
        while x_max >= 1:
            # Find the next y where x_max decreases
            next_y = float('inf')
            for A, B, C in ABC:
                if A == 0:
                    continue
                numerator = C + B -2 - x_max * A
                next_y_i = (numerator // B) + 1 if numerator % B != 0 else (numerator // B)
                if next_y_i < next_y:
                    next_y = next_y_i
            # Update total with the sum over this range
            dy = next_y - y
            if dy > 0:
                total += dy * x_max
            # Update y
            y = next_y
            # Compute new x_max
            x_max = min((C - B*y -1) // A for A, B, C in ABC if A*y + B*1 < C)
        
        print(total)

if __name__ == "__main__":
    main()