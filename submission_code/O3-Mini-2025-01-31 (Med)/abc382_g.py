def main():
    import sys,math
    data = sys.stdin.read().strip().split()
    if not data: 
        return
    t = int(data[0])
    # We'll use an index pointer for reading the numbers.
    ptr = 1
    outLines = []
    # Note: For negative division use math.floor(…) which is the real floor.
    for _ in range(t):
        K = int(data[ptr]); ptr+=1
        Sx = int(data[ptr]); ptr+=1
        Sy = int(data[ptr]); ptr+=1
        Tx = int(data[ptr]); ptr+=1
        Ty = int(data[ptr]); ptr+=1
        # Compute “shifted” coordinates: (point + 0.5)
        Xs = Sx + 0.5
        Ys = Sy + 0.5
        Xt = Tx + 0.5
        Yt = Ty + 0.5

        # For each point, define its cell indices:
        # (We use math.floor; note that math.floor returns the integer floor even for negatives.)
        iS = math.floor(Xs / K)
        jS = math.floor(Ys / K)
        iT = math.floor(Xt / K)
        jT = math.floor(Yt / K)
 
        # Determine tile “type” (by the parity of i+j):
        # If (i+j) is even, then type = H; else type = V.
        typeS = ((iS + jS) % 2 == 0)
        typeT = ((iT + jT) % 2 == 0)
 
        # Now compute the offset k.
        # For a horizontal tile (type H) the tile covers 
        #  x in [i*K, (i+1)*K] and y in [j*K + k, j*K + k + 1],
        # so we take k = floor(Y) – j*K.
        # For a vertical tile (type V) the tile covers 
        #  x in [i*K + k, i*K + k + 1] and y in [j*K, (j+1)*K],
        # so we take k = floor(X) – i*K.
        if typeS:
            kS = math.floor(Ys) - jS * K
        else:
            kS = math.floor(Xs) - iS * K
        if typeT:
            kT = math.floor(Yt) - jT * K
        else:
            kT = math.floor(Xt) - iT * K
 
        # Let D be the Manhattan distance between cells:
        D = abs(iS - iT) + abs(jS - jT)
 
        # now – by the formula – if D==0, answer is just |kS - kT|
        if D == 0:
            ans = abs(kS - kT)
        # When the cells share an edge, one may choose the free–move so that
        # the answer equals 1.
        elif D == 1:
            ans = 1
        else:
            # According to the formula:
            # when the cell types are the same the extra offset cost is 0,
            # while when they differ the extra cost is 
            #   extra = min(kS + kT, abs((kS + kT) - (K-1)) ).
            if typeS == typeT:
                ans = D
            else:
                extra = min(kS + kT, abs((kS + kT) - (K - 1)))
                ans = D + extra
        outLines.append(str(ans))
    sys.stdout.write("
".join(outLines))
 
if __name__ == '__main__':
    main()