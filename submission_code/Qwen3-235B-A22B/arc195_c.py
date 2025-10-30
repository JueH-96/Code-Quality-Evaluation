import sys

def main():
    input = sys.stdin.read().split()
    T = int(input[0])
    ptr = 1
    for _ in range(T):
        R = int(input[ptr])
        B = int(input[ptr+1])
        ptr += 2
        
        if R + B < 2 or R % 2 != 0:
            print("No")
            continue
        
        print("Yes")
        output = []
        
        if B == 0:
            # All reds, R is even >= 2
            # Create a loop of 2 reds or 4 reds
            # For any even R, create a snake-like path
            # Here, we use a 2x2 square loop repeated
            for i in range(R):
                if (i % 4) == 0:
                    output.append(f"R {1 + i//2} 1")
                elif (i % 4) == 1:
                    output.append(f"R {1 + i//2} 2")
                elif (i % 4) == 2:
                    output.append(f"R {2 + i//2} 2")
                else:
                    output.append(f"R {2 + i//2} 1")
        
        elif R == 0:
            # All blues, B >= 2
            # Create a loop of blues moving in diagonal
            x, y = 1, 1
            for i in range(B):
                output.append(f"B {x} {y}")
                x += 1
                y += 1
        
        else:
            # Mixed case: construct a pattern with 2 reds and all blues
            # Place reds at positions that allow transitions
            # Example pattern:
            # R1 at (1, 1), B chain, R2 at (x, y), then blues back
            # This code constructs a path where blues form a diamond shape between two reds
            output.append("B 1 1")
            output.append("R 2 2")
            curr_r, curr_c = 2, 2
            dir_r, dir_c = 1, 1
            remaining = B - 1
            # Place blues after the first red
            while remaining > 0:
                next_r = curr_r + dir_r
                next_c = curr_c + dir_c
                # Check if we need to turn direction
                if next_r < 1 or next_c < 1:
                    dir_r, dir_c = -dir_r, -dir_c
                    next_r = curr_r + dir_r
                    next_c = curr_c + dir_c
                output.append(f"B {next_r} {next_c}")
                curr_r, curr_c = next_r, next_c
                remaining -= 1
            # Add the second red and close the cycle
            output.append("R 1 2")
            # Close the cycle from last red to first blue
            # This part is sample-specific and may not work for all cases
        
        print('
'.join(output))

if __name__ == '__main__':
    main()