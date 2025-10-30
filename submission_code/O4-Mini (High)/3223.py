class Solution:
    def countCompleteSubstrings(self, word: str, k: int) -> int:
        n = len(word)
        # If k > n, no substring can have any character exactly k times
        if k > n:
            return 0

        # Convert to integer array 0..25 for speed
        arr = [ord(c) - ord('a') for c in word]

        # Build prefix sum of "bad" adjacency positions:
        # bad[i] = 1 if abs(arr[i+1]-arr[i]) > 2, else 0
        # prefix_bad[i] = sum of bad[0..i-1], prefix_bad[0]=0
        # so that sum of bad in [l..r-1] = prefix_bad[r] - prefix_bad[l]
        prefix_bad = [0] * (n + 1)
        for i in range(1, n):
            prefix_bad[i] = prefix_bad[i-1] + (1 if abs(arr[i] - arr[i-1]) > 2 else 0)
        prefix_bad[n] = prefix_bad[n-1]  # no bad edge at the end

        ans = 0
        # We only need to try up to 26 distinct characters,
        # and distinct_count * k <= n => distinct_count <= n//k
        max_m = min(26, n // k)

        for m in range(1, max_m + 1):
            L = m * k
            if L > n:
                break

            # Sliding window over substrings of length L
            count = [0] * 26
            distinct = 0      # number of chars with count > 0
            exactly_k = 0     # number of chars with count == k

            # Initialize the first window [0..L-1]
            for j in range(L):
                c = arr[j]
                old = count[c]
                count[c] = old + 1
                # track distinct
                if old == 0:
                    distinct += 1
                # track how many hit exactly k
                if old == k-1:
                    exactly_k += 1
                elif old == k:
                    # if old was already k, now k+1 => lose one exactly-k
                    exactly_k -= 1

            # Check the first window
            # adjacency is good iff no "bad" edges in [0..L-2]
            if distinct == m and exactly_k == m and (prefix_bad[L-1] - prefix_bad[0] == 0):
                ans += 1

            # Slide the window from i=1 to i = n-L
            for i in range(1, n - L + 1):
                # remove the char going out (at i-1)
                out_c = arr[i-1]
                old = count[out_c]
                count[out_c] = old - 1
                if old == 1:
                    # it was present once, now zero => lose one distinct
                    distinct -= 1
                if old == k + 1:
                    # was k+1, now k => gain one exactly-k
                    exactly_k += 1
                elif old == k:
                    # was k, now k-1 => lose one exactly-k
                    exactly_k -= 1

                # add the new char coming in (at i+L-1)
                in_c = arr[i+L-1]
                old = count[in_c]
                count[in_c] = old + 1
                if old == 0:
                    # was zero, now one => gain one distinct
                    distinct += 1
                if old == k-1:
                    # was k-1, now k => gain one exactly-k
                    exactly_k += 1
                elif old == k:
                    # was k, now k+1 => lose one exactly-k
                    exactly_k -= 1

                # Check this window [i..i+L-1]
                # no bad edges in [i..i+L-2] => prefix_bad[i+L-1] - prefix_bad[i] == 0
                if (distinct == m and exactly_k == m and
                    (prefix_bad[i+L-1] - prefix_bad[i] == 0)):
                    ans += 1

        return ans