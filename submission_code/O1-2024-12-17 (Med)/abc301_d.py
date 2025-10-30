def main():
    import sys
    S = sys.stdin.readline().strip()
    N = int(sys.stdin.readline().strip())
    
    # Convert N to binary string (no leading zeros).
    binN = bin(N)[2:]
    lenS = len(S)
    lenN = len(binN)
    
    # If N has more bits than S, then any number that S can form
    # (at most 2^lenS - 1) is definitely ≤ N. So we just fill all '?' with '1'
    # to get the maximum possible value.
    if lenN > lenS:
        candidate = ''.join('1' if c == '?' else c for c in S)
        print(int(candidate, 2))
        return
    
    # Otherwise, we do a digit-by-digit DP to ensure we stay ≤ N.
    # If lenN < lenS, pad binN with leading zeros to match lengths.
    if lenN < lenS:
        binN = binN.zfill(lenS)
    # Now len(binN) == lenS.
    
    # Memo for dp(pos, less): returns the best binary suffix from pos onward
    # or None if no valid assignment is possible.
    memo = {}
    
    def dp(pos, less):
        # If we've assigned all bits, return an empty string as the suffix.
        if pos == lenS:
            return ""
        if (pos, less) in memo:
            return memo[(pos, less)]
        
        bitS = S[pos]
        bitN = int(binN[pos])  # The bit of N at this position
        
        # If S[pos] is '0' or '1', we have no choice.
        if bitS in ('0', '1'):
            val = int(bitS)
            # If we haven't gone less yet, we can't exceed bitN here.
            if not less and val > bitN:
                memo[(pos, less)] = None
                return None
            new_less = less or (val < bitN)
            suffix = dp(pos+1, new_less)
            if suffix is None:
                memo[(pos, less)] = None
            else:
                memo[(pos, less)] = bitS + suffix
            return memo[(pos, less)]
        
        # If S[pos] == '?', we can try 1 or 0 (in descending order, to get max).
        else:
            # Try '1' first
            for val in [1, 0]:
                if not less and val > bitN:
                    continue  # can't exceed the bit of N
                new_less = less or (val < bitN)
                suffix = dp(pos+1, new_less)
                if suffix is not None:
                    memo[(pos, less)] = str(val) + suffix
                    return memo[(pos, less)]
            # If neither choice worked, no valid assignment
            memo[(pos, less)] = None
            return None
    
    result = dp(0, False)
    if result is None:
        print(-1)
    else:
        print(int(result, 2))

# Do not forget to call main()
main()