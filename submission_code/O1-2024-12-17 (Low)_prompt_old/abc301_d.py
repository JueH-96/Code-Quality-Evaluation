def solve():
    import sys
    data = sys.stdin.read().strip().split()
    S = data[0]
    N = int(data[1])
    
    # Convert N to binary (as a string)
    binN = bin(N)[2:]
    
    # We will use a digit-DP approach where we align S and binN to the same length
    # by padding with leading '0's, then we try to fill S so that it is as large
    # as possible but still ≤ N.
    
    L = max(len(S), len(binN))
    # Pad S and binN with leading zeros to make their lengths equal to L
    sS = S.rjust(L, '0')
    sN = binN.rjust(L, '0')
    
    # Memo for DP: dp(i, less) -> string of the best fill from i..L-1, or None if impossible
    # less = 0 means so far we are equal to prefix of sN, less = 1 means we've already been strictly less
    memo = {}
    
    def dp(i, less):
        # Base case: if we have assigned all bits
        if i == L:
            return ""  # no more bits to assign
        
        if (i, less) in memo:
            return memo[(i, less)]
        
        sc = sS[i]  # character in S
        nc = sN[i]  # character in binN
        
        # We want to pick the bit (0 or 1) as large as possible first
        # because we want the final number to be as large as possible
        candidates = ['1', '0'] if sc == '?' else [sc]
        
        best = None
        for bit in candidates:
            # Check if we are allowed to pick this bit
            b = int(bit)
            nbit = int(nc)
            
            # If we have not been strictly less yet, we cannot exceed nbit
            if less == 0 and b > nbit:
                continue
            
            # Compute nextLess
            if less == 1:
                nextLess = 1
            else:
                # less == 0
                if b < nbit:
                    nextLess = 1
                else:
                    nextLess = 0
            
            # Recursively fill the rest
            res = dp(i+1, nextLess)
            if res is not None:
                candidate_str = bit + res
                if best is None:
                    best = candidate_str
                else:
                    # Compare to see which is larger lexicographically
                    # but we pick the first valid in descending order,
                    # so that should already be the best
                    pass
                # Since we are checking bits in descending order, the first fit is the best
                break
        
        memo[(i, less)] = best
        return best
    
    ans_bin = dp(0, 0)
    
    if ans_bin is None:
        print(-1)
    else:
        # Convert the resulting binary string to decimal
        # (it may have leading zeros, but that doesn't affect the integer value)
        val = int(ans_bin, 2)
        # We must verify it's indeed ≤ N (it should be by construction)
        if val <= N:
            print(val)
        else:
            print(-1)

def main():
    solve()

if __name__ == "__main__":
    main()