def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    A = int(data[1])
    B = int(data[2])
    D = list(map(int, data[3:3+N]))
    
    MOD = A + B
    
    # For each D_i, compute L_i and R_i
    L = []
    R = []
    for d in D:
        rem = d % MOD
        Li = (MOD - rem) % MOD
        Ri = (Li + A - 1) % MOD
        L.append(Li)
        R.append(Ri)
    
    # Find max_L and min_R
    max_L = max(L)
    min_R = min(R)
    
    # Check if there is an intersection
    if max_L <= min_R:
        # Intersection is [max_L, min_R]
        print("Yes")
    else:
        # Intersection is [min_R, MOD-1] and [0, max_L-1]
        # Need to check if these intervals are non-empty
        if min_R < MOD and max_L < MOD:
            print("Yes")
        else:
            print("No")

if __name__ == "__main__":
    main()