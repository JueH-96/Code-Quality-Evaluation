def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    t = int(input_data[0])
    ptr = 1
    
    # We will process each test in O(1) time (aside from a possible O(|X|) check if needed).
    # Let S be a known string of length s.  Let X, Y be bit‐strings of lengths mX, mY.
    # Count a = number of '0' in X, b = number of '1' in X.
    #       c = number of '0' in Y, d = number of '1' in Y.
    #
    # We want f(S,T,X) == f(S,T,Y) for some (possibly empty) T.
    #
    # Key observations:
    # 1) Let |S| = s, let |T| = t (unknown).
    # 2) The length of f(S,T,X) is a*s + b*t;  the length of f(S,T,Y) is c*s + d*t.
    #    For them even to have the same length, we must satisfy:
    #       a*s + b*t == c*s + d*t  =>  (a-c)*s + (b-d)*t = 0.
    #
    # Case A: If b == d:
    #   Then the length equation reduces to  (a-c)*s = 0. Since s>0, we need a == c.
    #   - If a != c => No.
    #   - Otherwise => Yes (we can pick any t, for instance t=0).
    #
    # Case B: If b != d:
    #   Solve for t = ((c - a)*s) / (b - d).  This must be an integer >= 0.
    #   If not integer or < 0 => No.
    #
    #   Let that integer be Tlen.
    #   Now we must check if the block‐by‐block concatenation can match.
    #   It turns out all the tricky "cross‐matching" conditions boil down to:
    #   - T can be empty (t=0) => then the final string is just S repeated
    #       for each '0' in X or Y.  For them to match, we need a == c.
    #   - T can be exactly S (t=s) => then effectively every block is S;
    #       we need the total # of blocks to match => (a+b) == (c+d).
    #   - Otherwise, if t not in {0, s}, then the only way to avoid forcing
    #       T = S (or contradiction) is that whenever X has '0', Y also has '0'
    #       at the corresponding “block” position in the final expansion,
    #       and likewise for '1'↔'1', so effectively X must be exactly Y
    #       (same bit‐string).  Then the expansions line up block‐by‐block
    #       with no cross‐type alignment.  So in that scenario we check X==Y.
    #
    # Putting it all together:
    #
    # 1) Compute a,b,c,d.
    # 2) If b == d:
    #      if a == c => "Yes" else => "No".
    #    else:
    #      Tlen = ((c - a)*s) / (b - d)  (must be integer >= 0)
    #      if not valid => "No"
    #      else:
    #        if Tlen == 0 => "Yes" iff a==c else "No"
    #        elif Tlen == s => "Yes" iff (a+b)==(c+d) else "No"
    #        else => "Yes" iff X==Y else "No"
    #
    # This completes an O(1) decision per test, except for the possible check X==Y,
    # which is O(|X|).  However, the total sum of |X| (and |Y|) over all tests
    # does not exceed 5e5, so in total this will be efficient enough.

    out = []
    # Indexes in input_data: each case has S, X, Y
    # so for each test we read 3 lines from input_data
    for _ in range(t):
        S = input_data[ptr]; ptr += 1
        X = input_data[ptr]; ptr += 1
        Y = input_data[ptr]; ptr += 1
        
        s = len(S)
        a = X.count('0')
        b = len(X) - a
        c = Y.count('0')
        d = len(Y) - c
        
        # Case A: b == d
        if b == d:
            # Then we need a == c
            if a == c:
                out.append("Yes")
            else:
                out.append("No")
            continue
        
        # Case B: b != d
        diff = (c - a)*s
        denom = (b - d)
        # check integer solution
        if denom == 0:
            # Should never happen here since b != d, but just in case:
            out.append("No")
            continue
        
        if diff % denom != 0:
            # not integer
            out.append("No")
            continue
        
        Tlen = diff // denom
        if Tlen < 0:
            out.append("No")
            continue
        
        # Now handle the subcases
        if Tlen == 0:
            # Then the expansions only come from S for '0' bits
            # Must have a == c
            if a == c:
                out.append("Yes")
            else:
                out.append("No")
        elif Tlen == s:
            # Then T is effectively S
            # expansions each produce (a+b) or (c+d) copies of S
            if (a + b) == (c + d):
                out.append("Yes")
            else:
                out.append("No")
        else:
            # T has some other length
            # Must avoid cross matching => X must be exactly Y
            # Also length of X and Y must match if they are to be equal
            if len(X) == len(Y) and X == Y:
                out.append("Yes")
            else:
                out.append("No")
    
    # Print all results
    print("
".join(out))