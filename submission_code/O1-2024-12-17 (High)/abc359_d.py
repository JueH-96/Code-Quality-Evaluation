def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    K = int(data[1])
    S = data[2]
    mod = 998244353
    
    # Quick way to map a character 'A','B','?' to possible bits (0 for 'A', 1 for 'B')
    def getPossibleBits(ch):
        if ch == 'A':
            return [0]
        elif ch == 'B':
            return [1]
        else:
            return [0, 1]  # '?' can be either
    
    # When K <= 1 it would be trivial, but constraints say K >= 2.
    # Precompute whether the length-K substring (represented by (mask,cBit)) is a palindrome
    # mask holds the last (K-1) bits, cBit is the new bit, so total K bits to check.
    def build_is_pal(K):
        # Helper to get the bit at position pos in the length-K substring
        # pos=0 is leftmost, pos=K-1 is rightmost
        def subPos(pos, mask, cBit):
            if pos == K - 1:
                return cBit
            else:
                # The bit for pos < (K-1) is stored in mask, leftmost bit is mask >> (K-2)
                return (mask >> (K - 2 - pos)) & 1
        
        L = 1 << (K - 1)
        is_pal = [[False]*2 for _ in range(L)]
        half = K // 2
        for mask in range(L):
            for bit in (0, 1):
                pal = True
                for j in range(half):
                    if subPos(j, mask, bit) != subPos(K - 1 - j, mask, bit):
                        pal = False
                        break
                is_pal[mask][bit] = pal
        return is_pal

    # Build the palindrome lookup table
    is_pal = build_is_pal(K)
    
    # We will use a rolling DP of size 2^(K-1). dp[i] = number of ways to form
    # assignments for T[0..i-1] with a given mask of last K-1 bits.
    # Initially, dp[0][0] = 1 (no characters assigned, "empty" mask).
    L = 1 << (K - 1)  # size of DP dimension for the mask
    dp_current = [0] * L
    dp_current[0] = 1
    
    for i in range(N):
        dp_next = [0] * L
        possible_bits = getPossibleBits(S[i])
        
        for mask in range(L):
            ways = dp_current[mask]
            if ways == 0:
                continue
            
            for cBit in possible_bits:
                # If we already have at least K-1 chars assigned, check if adding this one
                # forms a palindromic substring of length K
                if i >= K - 1 and is_pal[mask][cBit]:
                    # This new length-K block would be a palindrome, skip
                    continue
                
                # Otherwise, we can transition
                new_mask = ((mask << 1) & (L - 1)) | cBit
                dp_next[new_mask] = (dp_next[new_mask] + ways) % mod
        
        dp_current = dp_next
    
    # The result is the sum of ways over all possible masks after assigning all N chars
    answer = sum(dp_current) % mod
    print(answer)

# Do not forget to call main()
main()