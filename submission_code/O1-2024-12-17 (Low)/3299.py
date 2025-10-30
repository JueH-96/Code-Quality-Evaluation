class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        import collections

        freq = collections.Counter(nums)
        # Edge case: If there's only one distinct value overall,
        # answer can be either the total count (if it's odd) or count-1 (if even), for that single value.
        # However, we'll handle this more generally below.

        # First, handle the special case x=1 separately:
        # For x=1, 1^(2^k) = 1 for all k, so to form a valid symmetric pattern of length L = 2k+1,
        # we need exactly L copies of 1. The longest odd length ≤ freq[1].
        max_len = 1  # At least we can always pick a single element from the array if not empty
        if 1 in freq:
            c = freq[1]
            if c >= 1:
                # The largest odd integer <= c is c if c is odd, else c-1
                max_len = max(max_len, c if c % 2 == 1 else c - 1)

        # Now gather all unique bases except 1
        unique_vals = list(freq.keys())
        # We'll process x=1 separately, so skip it here
        if 1 in unique_vals:
            unique_vals.remove(1)

        # For each x > 1, we try to build the pattern x, x^2, x^4, ...
        # The pattern length for "k" steps is (2*k + 1),
        # requiring freq[x^(2^k)] >= 1 and freq[x^(2^i)] >= 2 for i < k.
        # We'll generate powers of x by repeated squaring until they exceed 10^9 (the problem constraint).
        for x in unique_vals:
            # Quick check: if freq[x] < 1, skip
            # (Normally won't happen since x is from freq keys, but just to be safe.)
            if freq[x] < 1:
                continue

            # Collect powers of x by repeated squaring
            pows = [x]
            while pows[-1] <= 10**9 and pows[-1] <= 10**9 // pows[-1]:
                nxt = pows[-1] * pows[-1]
                if nxt > 10**9:
                    break
                pows.append(nxt)

            # Build an array of frequencies for these powers
            freq_arr = [freq[val] for val in pows]

            # partial = k means we can form a pattern of length 2*k + 1
            # Start with k = 0: needs freq[x] >= 1
            if freq_arr[0] < 1:
                # can't even form length=1 with x
                continue

            partial = 0  # we can form length=1
            min2 = freq_arr[0]  # track the minimum freq among pows[0..k-1] as we go

            # Try to extend k from 1 up to len(pows)-1
            for k in range(1, len(pows)):
                # We need freq_arr[k] >= 1 for x^(2^k),
                # and freq_arr[0..k-1] >= 2 each; in practice, we track min2 so far
                if freq_arr[k] < 1:
                    break

                # Update min2 to also include freq_arr[k-1]
                min2 = min(min2, freq_arr[k-1])
                if min2 < 2:
                    # can't extend further
                    break

                # If we get here, partial = k is valid
                partial = k

            # partial means we can form pattern of length 2*partial + 1
            length = 2 * partial + 1
            if length > max_len:
                max_len = length

        return max_len