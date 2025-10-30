import sys

def main():
    input = sys.stdin.read().split()
    idx = 0
    T = int(input[idx])
    idx += 1
    for _ in range(T):
        K = int(input[idx])
        Sx = int(input[idx+1])
        Sy = int(input[idx+2])
        Tx = int(input[idx+3])
        Ty = int(input[idx+4])
        idx +=5
        
        # Compute i_start and j_start
        i_start = (2 * Sx + 1) // (2 * K)
        j_start = (2 * Sy + 1) // (2 * K)
        i_end = (2 * Tx + 1) // (2 * K)
        j_end = (2 * Ty + 1) // (2 * K)
        
        distance = abs(i_end - i_start) + abs(j_end - j_start)
        
        # Check parity
        parity_start = (i_start % 2) == (j_start % 2)
        parity_end = (i_end % 2) == (j_end % 2)
        if parity_start != parity_end:
            distance += 1
        
        print(distance)
        
if __name__ == "__main__":
    main()