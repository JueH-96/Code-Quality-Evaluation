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
        
        i_start = Sx // K
        j_start = Sy // K
        i_target = Tx // K
        j_target = Ty // K
        
        dx = abs(i_target - i_start)
        dy = abs(j_target - j_start)
        steps = dx + dy
        
        p_start = (i_start + j_start) % 2
        p_target = (i_target + j_target) % 2
        
        if p_start != p_target:
            steps += 1
        
        print(steps)
        
if __name__ == "__main__":
    main()