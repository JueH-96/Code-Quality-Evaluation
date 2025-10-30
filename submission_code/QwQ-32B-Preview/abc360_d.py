import sys
import bisect

def main():
    # Read input
    N_T = sys.stdin.readline().split()
    N = int(N_T[0])
    T = int(N_T[1])
    S = sys.stdin.readline().strip()
    X = list(map(int, sys.stdin.readline().split()))
    
    # Scale T and X by 10 to handle floating points
    T_scaled = T * 10 + 1  # 2*(T + 0.1) becomes 2*T + 2 when T is scaled by 10
    X_scaled = [x * 10 for x in X]
    
    # Create list of (position, direction)
    ants = sorted(zip(X_scaled, S))
    
    # Extract positions of ants facing left
    L = [x for x, s in ants if s == '0']
    
    count = 0
    for x, s in ants:
        if s == '1':
            # Calculate upper limit for x_j
            upper_limit = x + 2 * T_scaled
            # Count ants facing left with x_j <= upper_limit
            count += bisect.bisect_right(L, upper_limit)
    
    print(count)

if __name__ == '__main__':
    main()