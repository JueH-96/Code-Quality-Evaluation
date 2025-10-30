class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        import collections
        
        freq = collections.Counter(nums)
        
        # Handle the special case x = 1 separately to avoid infinite loops,
        # since 1^(2^k) = 1 for all k, and we can use as many "1"s as we have,
        # but the pattern length must be odd: 2*m + 1.
        # Thus the maximum we can use is the largest odd number <= freq(1).
        answer = 0
        if 1 in freq:
            count_ones = freq[1]
            if count_ones % 2 == 1:
                answer = count_ones
            else:
                answer = count_ones - 1
        
        # Collect all distinct values except 1 (already handled)
        unique_vals = sorted([x for x in freq.keys() if x != 1])
        max_val_in_array = max(freq.keys())
        
        # We will try each distinct x > 1 as the base of the pattern.
        for x in unique_vals:
            # If frequency is zero (shouldn't normally happen unless we modified freq, but be safe)
            if freq[x] < 1:
                continue
            
            # Build a list of frequencies for powers: x^(2^0), x^(2^1), ...
            # Stop once we exceed max_val_in_array or 10^9, or if x == 1 (handled above).
            pow_freqs = []
            cur = x
            while cur <= max_val_in_array and cur <= 10**9:
                pow_freqs.append(freq.get(cur, 0))
                # Compute next power by squaring, but check for overflow or no growth.
                # If cur * cur > 10^9 or cur * cur > max_val_in_array, we can stop.
                nxt = cur * cur
                if nxt > 10**9 or nxt > max_val_in_array or nxt <= cur:
                    break
                cur = nxt
            
            # Now pow_freqs[i] = how many times x^(2^i) appears in nums (or 0 if not at all).
            # We'll check from i=0 upwards. For a center i, we need pow_freqs[i] >= 1,
            # and for all j < i, we need pow_freqs[j] >= 2.
            
            pairs_ok = True  # Will track if for all j < i, pow_freqs[j] >= 2
            max_pattern_for_x = 0
            
            for i in range(len(pow_freqs)):
                if i == 0:
                    # No pairs needed yet
                    pairs_ok = True
                else:
                    # We require pow_freqs[i-1] >= 2 to keep pairs_ok
                    pairs_ok = pairs_ok and (pow_freqs[i-1] >= 2)
                
                # If we can place a center at i, check length = 2*i + 1
                if pow_freqs[i] >= 1 and pairs_ok:
                    length = 2*i + 1
                    if length > max_pattern_for_x:
                        max_pattern_for_x = length
            
            # Update the global answer
            answer = max(answer, max_pattern_for_x)
        
        return answer