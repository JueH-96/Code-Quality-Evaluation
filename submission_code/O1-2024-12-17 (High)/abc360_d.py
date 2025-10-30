def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    
    N = int(input_data[0])
    T = int(input_data[1])
    S = input_data[2]
    X = list(map(int, input_data[3:]))

    # Separate positions into those facing right (1) and those facing left (0)
    Rpos = []
    Lpos = []
    for i in range(N):
        if S[i] == '1':
            Rpos.append(X[i])
        else:
            Lpos.append(X[i])
    
    # Sort positions
    Rpos.sort()
    Lpos.sort()
    
    # We use the fact that crossing occurs if and only if:
    #   (antR is to the left and facing right) and (antL is to the right and facing left)
    #   and the distance between them <= 2*T.
    # This is because the crossing time = (xL - xR)/2, which must be < T+0.1.
    # For integer positions, (xL - xR) <= 2*T guarantees crossing time <= T < T+0.1.
    limitDist = 2 * T
    
    ans = 0
    j = 0  # pointer into Lpos
    k = -1 # will track the maximal index where Lpos[k] <= r + limitDist
    nL = len(Lpos)
    
    # Two-pointer sweep over Rpos (each r in ascending order)
    for r in Rpos:
        # Move j forward so that Lpos[j] > r (we need xL > xR to possibly cross)
        while j < nL and Lpos[j] <= r:
            j += 1
        
        # Move k forward so that Lpos[k] <= r + limitDist
        # (largest k with Lpos[k] <= r+limitDist)
        while k + 1 < nL and Lpos[k + 1] <= r + limitDist:
            k += 1
        
        # Now, all indices from j to k are within (r, r + limitDist],
        # so they form valid crossing pairs with this r
        if k >= j:
            ans += (k - j + 1)
    
    print(ans)

# Do not forget to call main()
main()