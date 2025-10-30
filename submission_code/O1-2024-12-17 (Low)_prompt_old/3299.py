class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        """
        We want to find the largest subset of nums whose elements can be arranged
        in a palindromic 'square' pattern:

            [x, x^2, x^4, ..., x^(2^k), ..., x^4, x^2, x]

        - For k=0, the pattern is just [x].
        - For k>0, the pattern has length 2*k + 1:
            x^(2^0) appears 2 times, x^(2^1) appears 2 times, ... , x^(2^(k-1)) appears 2 times,
            x^(2^k) appears 1 time.

        That translates to these frequency requirements:
            - If k=0, freq[x] >= 1
            - If k=1, freq[x] >= 2, freq[x^2] >= 1
            - If k=2, freq[x] >= 2, freq[x^2] >= 2, freq[x^4] >= 1
            - And so on:
              for i in [0..k-1], freq[x^(2^i)] >= 2,
              and freq[x^(2^k)] >= 1.

        We want the maximum possible length of such a pattern from a subset of nums.

        Approach:
        1. Count the frequency of each value in nums.
        2. Handle base = 1 as a special case (powers of 1 remain 1):
           - If freq(1) = f, then we can form a pattern of up to the largest odd number ≤ f.
             Because for k>0 we need 2*k+1 copies of "1".
             So max length = 2 * floor((f-1) / 2) + 1.
        3. For each distinct base x > 1, build a list of frequencies of consecutive powers:
           freqList[0] = freq[x]
           freqList[1] = freq[x^2]
           freqList[2] = freq[x^4]
           ...
           Stop if x^(2^i) > 10^9 or if no more powers are relevant.
        4. Check the largest k we can achieve:
           - k=0 if freqList[0] >= 1  => pattern length = 1
           - for larger k = i (i >= 1),
             need all freqList[0..i-1] >= 2 and freqList[i] >= 1 => pattern length = 2*i + 1
           We can do this efficiently by keeping a running minimum of freqList[0..i-1]
           to quickly check if it is >= 2.
        5. Take the maximum across all bases.

        Time complexity analysis:
        - Counting frequencies: O(n).
        - Let D be the number of distinct values. Typically D <= n.
        - For each distinct value x (> 1), we generate powers up to ~ log2(1e9) ~= 30
          and do a pass of length ~ 30. That yields about O(D * 30) operations.
        - This is feasible for n up to 1e5.

        We'll return the maximum pattern length found.
        """

        from collections import Counter
        freq = Counter(nums)

        # Special handling for x = 1
        best = 0
        if freq[1] > 0:
            f = freq[1]
            # Largest odd number <= f
            # This is 2 * floor((f - 1)/2) + 1
            best = 2 * ((f - 1) // 2) + 1

        # We'll check other bases > 1
        # Powers can get large, but Python can handle big integers;
        # we just stop when we exceed 1e9 (since nums[i] <= 1e9).
        for x in list(freq.keys()):
            # Skip 1 (already handled) or zero-frequency
            if x == 1 or freq[x] == 0:
                continue

            # freqList[i] = frequency of x^(2^i), stopping when exceeding 1e9
            val = x
            if val > 10**9:
                continue  # x^(2^0)=x is already too big (though x <= 1e9 by problem statement, so typically won't hit)
            freqList = [freq[val]]

            # Build powers
            while val <= 10**9:
                val = val * val
                if val > 10**9:
                    break
                freqList.append(freq.get(val, 0))

            # Now find the largest k we can achieve
            # k=0 => need freqList[0]>=1 => pattern length=1
            # k=1 => need freqList[0]>=2, freqList[1]>=1 => pattern length=3
            # ...
            # General k => for i in [0..k-1], freqList[i]>=2, and freqList[k]>=1 => pattern=2*k+1

            # If freqList[0] < 1, no pattern possible from this base
            if freqList[0] < 1:
                continue

            # localK will store the best k we can get for this base
            # localK >= 0 => means at least we can do k=0 => pattern length=1
            localK = 0

            # We'll keep track of a running "min2upTo[i]" = min(freqList[0..i])
            # so we can quickly check if freqList[0..i]>=2
            min2upTo = [0] * len(freqList)
            min2upTo[0] = freqList[0]

            for i in range(1, len(freqList)):
                min2upTo[i] = min(min2upTo[i-1], freqList[i])

            # We already know freqList[0] >= 1 => pattern=1
            best_for_this_x = 1  # pattern length so far

            # Now check for k=1.. up to len(freqList)-1
            # Condition: for i in [0..(k-1)], freqList[i]>=2 => min2upTo[k-1]>=2
            #            and freqList[k]>=1
            for k in range(1, len(freqList)):
                # Check if min of freqList[0..k-1] >= 2 and freqList[k] >= 1
                if min2upTo[k-1] >= 2 and freqList[k] >= 1:
                    # Then pattern length = 2*k + 1
                    cand_len = 2 * k + 1
                    if cand_len > best_for_this_x:
                        best_for_this_x = cand_len
                else:
                    # If we fail for k, higher k' > k won't be possible
                    # because we also need freqList[k]>=2 for k' > k-1, etc.
                    break

            best = max(best, best_for_this_x)

        return best