class Solution:
    def makeStringGood(self, s: str) -> int:
        # Count frequency for each letter a-z
        cnt = [0] * 26
        for ch in s:
            cnt[ord(ch) - ord('a')] += 1

        n = len(s)
        total_best = float('inf')
        prefix = [0] * 27  # prefix sum for counts
        for i in range(26):
            prefix[i+1] = prefix[i] + cnt[i]

        # Helper: cost for outside segment = letters not in [L,R]
        def outside_cost(L, R):
            return prefix[L] + (prefix[26] - prefix[R+1])
        
        # We try every contiguous segment of the alphabet, from L to R.
        for L in range(26):
            for R in range(L, 26):
                k = R - L + 1  # number of letters we will have in final string.
                # For a chosen frequency f, final total length becomes f*k.
                # We try possible f: from 1 to n (n is max possible, insertion can be done)
                # (We could try f values near n//k, but trying all is simpler)
                base_outside = outside_cost(L, R)
                # We'll loop f from 1 to n maybe, but we can also stop if f*k becomes "very large".
                # f*k should be approximately around n (we pay for insertion or deletion difference).
                # To be safe we loop f from 1 to n. 
                for f in range(1, n+1):
                    cost_inner = 0
                    carry = 0
                    # Process letters in our segment from L to R.
                    for index in range(L, R+1):
                        # available from current letter after shifting in carry
                        available = carry + cnt[index]
                        if available >= f:
                            surplus = available - f
                            # We must shift surplus to the next letter, costing surplus (each one cost 1)
                            cost_inner += surplus
                            carry = surplus
                        else:
                            # shortage -> insertion needed.
                            shortage = f - available
                            cost_inner += shortage
                            carry = 0
                    # After processing the last letter, any leftover supply must be deleted.
                    cost_inner += carry
                    total_cost = base_outside + cost_inner
                    if total_cost < total_best:
                        total_best = total_cost
                    
                    # A small optimization: if f*k (final desired length) becomes much larger than n,
                    # then insertion cost will only increase. We can break early.
                    # Let current insertion cost estimate be at least f*k - n.
                    if f * k - n > total_best:
                        break

        return total_best

# Example usage:
if __name__ == '__main__':
    sol = Solution()
    print(sol.makeStringGood("acab"))   # Expected output 1
    print(sol.makeStringGood("wddw"))   # Expected output 0
    print(sol.makeStringGood("aaabc"))  # Expected output 2