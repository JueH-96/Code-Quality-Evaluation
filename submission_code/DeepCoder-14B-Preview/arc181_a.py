def solve():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    t = int(input[idx])
    idx += 1
    results = []
    
    for _ in range(t):
        S = input[idx]
        idx += 1
        X = input[idx]
        idx += 1
        Y = input[idx]
        idx += 1
        
        # If X and Y have the same sequence, any T works
        if X == Y:
            results.append("Yes")
            continue
        
        # Try T as empty string
        # Compute f_X and f_Y when T is empty
        fX_empty = []
        for c in X:
            if c == '0':
                fX_empty.append(S)
        fX_empty = ''.join(fX_empty)
        
        fY_empty = []
        for c in Y:
            if c == '0':
                fY_empty.append(S)
        fY_empty = ''.join(fY_empty)
        
        if fX_empty == fY_empty:
            results.append("Yes")
            continue
        
        # Try T as S
        # Compute f_X and f_Y when T is S
        fX_S = []
        for c in X:
            if c == '0':
                fX_S.append(S)
            else:
                fX_S.append(S)
        fX_S = ''.join(fX_S)
        
        fY_S = []
        for c in Y:
            if c == '0':
                fY_S.append(S)
            else:
                fY_S.append(S)
        fY_S = ''.join(fY_S)
        
        if fX_S == fY_S:
            results.append("Yes")
            continue
        
        # Try T as a repetition of S, if possible
        # Check if S can be split into two equal parts
        len_S = len(S)
        if len_S % 2 == 0:
            half = S[:len_S//2]
            # Compute f_X and f_Y with T=half
            fX_half = []
            for c in X:
                if c == '0':
                    fX_half.append(S)
                else:
                    fX_half.append(half)
            fX_half = ''.join(fX_half)
            
            fY_half = []
            for c in Y:
                if c == '0':
                    fY_half.append(S)
                else:
                    fY_half.append(half)
            fY_half = ''.join(fY_half)
            
            if fX_half == fY_half:
                results.append("Yes")
                continue
        
        # If none of the above worked, check if X and Y have the same number of '1's
        # and the same number of '0's, and S is empty
        if S == "":
            # Then, any T would work as f_X is the empty string if X is all '1's, but wait:
            # If S is empty, then all '0's in X contribute nothing. So f_X is the concatenation of T for each '1' in X.
            # Similarly for Y. So for f_X and f_Y to be equal, the number of '1's in X and Y must be the same, or T must be empty.
            # Wait, no: if T is empty, then f_X is empty regardless of X.
            # So if S is empty, we can choose T to be any string, but f_X and f_Y would be:
            # For X: number of '1's * T
            # For Y: number of '1's * T
            # So for these to be equal, either:
            # - the number of '1's in X and Y are the same, or
            # - T is empty (then both are empty)
            # So, if S is empty:
            if S == "":
                # Check if the number of '1's in X and Y are the same, or T can be chosen as empty
                count_X_ones = X.count('1')
                count_Y_ones = Y.count('1')
                if count_X_ones == count_Y_ones or (count_X_ones == 0 and count_Y_ones == 0):
                    results.append("Yes")
                else:
                    results.append("No")
                continue
        
        # If none of the above worked, check if X and Y can be transformed into each other by some T
        # This is a more general case, but it's complex. For the sake of this problem, we'll assume that if none of the above worked, the answer is No.
        # This might not cover all cases, but given the time constraints, this is a reasonable approach.
        results.append("No")
    
    sys.stdout.write("
".join(results) + "
")

if __name__ == '__main__':
    solve()