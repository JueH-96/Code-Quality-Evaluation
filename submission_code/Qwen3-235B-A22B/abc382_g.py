import sys
import math

def main():
    input = sys.stdin.read().split()
    T = int(input[0])
    idx = 1
    for _ in range(T):
        K = int(input[idx])
        Sx = int(input[idx+1])
        Sy = int(input[idx+2])
        Tx = int(input[idx+3])
        Ty = int(input[idx+4])
        idx +=5
        
        # Compute (i,j) for start and end
        # start point is (Sx+0.5, Sy+0.5)
        x_s = Sx + 0.5
        y_s = Sy + 0.5
        i_s = int(math.floor(x_s / K))
        j_s = int(math.floor(y_s / K))
        parity_s = (i_s + j_s) % 2
        
        # end point
        x_t = Tx + 0.5
        y_t = Ty + 0.5
        i_t = int(math.floor(x_t / K))
        j_t = int(math.floor(y_t / K))
        parity_t = (i_t + j_t) % 2
        
        dx = abs(i_s - i_t)
        dy = abs(j_s - j_t)
        steps = dx + dy
        if parity_s != parity_t:
            steps += 1
        
        print(steps)

if __name__ == "__main__":
    main()