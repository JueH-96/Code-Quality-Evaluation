import sys

def main():
    N, M = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    B = list(map(int, sys.stdin.readline().split()))
    
    total_steps = 0
    
    for i in range(N):
        a = A[i]
        b = B[i]
        if a == b:
            continue
        
        delta_plus = (b - a) % M
        delta_minus = (a - b) % M
        
        valid_dirs = []
        
        # Check clockwise direction
        valid = True
        # Check left neighbor (A[i-1])
        if i > 0:
            x = A[i-1]
            if b >= a:
                if a <= x <= b:
                    valid = False
            else:
                if x >= a or x <= b:
                    valid = False
        # Check right neighbor (A[i+1])
        if i < N-1:
            x = A[i+1]
            if b >= a:
                if a <= x <= b:
                    valid = False
            else:
                if x >= a or x <= b:
                    valid = False
        if valid:
            valid_dirs.append(delta_plus)
        
        # Check counter-clockwise direction
        valid = True
        if i > 0:
            x = A[i-1]
            if b <= a:
                if b <= x <= a:
                    valid = False
            else:
                if x >= b or x <= a:
                    valid = False
        if i < N-1:
            x = A[i+1]
            if b <= a:
                if b <= x <= a:
                    valid = False
            else:
                if x >= b or x <= a:
                    valid = False
        if valid:
            valid_dirs.append(delta_minus)
        
        if not valid_dirs:
            print(-1)
            return
        
        total_steps += min(valid_dirs)
    
    print(total_steps)

if __name__ == '__main__':
    main()