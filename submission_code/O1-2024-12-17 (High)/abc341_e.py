def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    
    # Fast extractor for tokens
    # We'll parse them in order: N, Q, S, then Q queries each of which has "1 L R" or "2 L R"
    idx = 0
    N = int(input_data[idx]); idx += 1
    Q = int(input_data[idx]); idx += 1
    S_str = input_data[idx]; idx += 1
    
    # Convert S into 1-based list of bits
    s = [0] * (N+1)
    for i in range(1, N+1):
        s[i] = 1 if S_str[i-1] == '1' else 0
    
    # -----------------------------
    # Fenwicks for parity flips (XOR Fenwicks).
    # We'll store a Fenwicks array that allows range flips L..R by:
    #   fenwXorUpdate(L, 1)
    #   fenwXorUpdate(R+1, 1)   (if R+1 <= N)
    # Then the value at position i is s[i] ^ fenwXorQuery(i).
    # Implementation in 1-based indexing for convenience.
    # -----------------------------
    flipFenw = [0]*(N+1)
    
    def fenwXorUpdate(i):
        # toggles (xor=1) in Fenwicks from i upward
        while i <= N:
            flipFenw[i] ^= 1
            i += (i & -i)
    
    def fenwXorQuery(i):
        # returns the parity (xor) from 1..i
        ret = 0
        while i > 0:
            ret ^= flipFenw[i]
            i -= (i & -i)
        return ret
    
    def getSVal(i):
        # current value of S[i] after flips
        return s[i] ^ fenwXorQuery(i)
    
    # -----------------------------
    # Fenwicks for counting how many adjacent pairs are different:
    # We'll store diff[i] = 1 if S[i] != S[i+1], else 0, for i=1..N-1.
    # Then a substring S[L..R] is "good" if sum(diff[L..R-1]) == (R-L).
    #
    # We maintain diff in an array so we can do (newDiff - oldDiff) updates in Fenwicks.
    # We'll keep a Fenwicks that supports point updates and prefix sum queries.
    # -----------------------------
    diff = [0]*(N+1)
    for i in range(1, N):
        diff[i] = 1 if s[i] != s[i+1] else 0
    
    # Build Fenwicks for sums in O(N) time
    fenwDiff = [0]*(N+1)
    # Copy diff[i] into fenwDiff
    for i in range(1, N):
        fenwDiff[i] = diff[i]
    # Build Fenwicks structure
    for i in range(1, N):
        j = i + (i & -i)
        if j <= N:
            fenwDiff[j] += fenwDiff[i]
    
    def fenwDiffSum(i):
        s_ = 0
        while i > 0:
            s_ += fenwDiff[i]
            i -= (i & -i)
        return s_
    
    def fenwDiffRangeSum(l, r):
        if r < l:
            return 0
        return fenwDiffSum(r) - fenwDiffSum(l-1)
    
    def fenwDiffPointAdd(i, val):
        # point update: diff[i] += val
        while i <= N:
            fenwDiff[i] += val
            i += (i & -i)
    
    def recalcDiff(i):
        # Recalculate diff[i] = 1 if getSVal(i)!= getSVal(i+1) else 0
        # Then update fenwDiff accordingly
        if i < 1 or i >= N:
            return
        oldVal = diff[i]
        newVal = 1 if getSVal(i) != getSVal(i+1) else 0
        diff[i] = newVal
        delta = newVal - oldVal
        if delta != 0:
            fenwDiffPointAdd(i, delta)
    
    # We'll collect answers for the type=2 queries
    out = []
    
    # Process queries
    # Each query is either "1 L R" or "2 L R"
    # We'll read them from input_data using idx
    for _ in range(Q):
        t = int(input_data[idx]); idx += 1
        L = int(input_data[idx]); idx += 1
        R = int(input_data[idx]); idx += 1
        
        if t == 1:
            # Flip each character from L..R
            # Fenwicks XOR range update
            fenwXorUpdate(L)
            if R+1 <= N:
                fenwXorUpdate(R+1)
            
            # Recalc boundaries: diff[L-1], diff[R] if valid
            recalcDiff(L-1)
            recalcDiff(R)
            
        else:
            # t == 2: Check if substring S[L..R] is good
            length = R - L
            # sum of diff[L..R-1] must be (R-L)
            d = fenwDiffRangeSum(L, R-1)
            if d == length:
                out.append("Yes")
            else:
                out.append("No")
    
    # Print all results
    print("
".join(out))

# Don't forget to call main()
if __name__ == "__main__":
    main()