class Solution:
    def maximumTotalDamage(self, power: List[int]) -> int:
        from bisect import bisect_left

        # Frequency map: damage -> how many spells have this damage
        freq = {}
        for dmg in power:
            freq[dmg] = freq.get(dmg, 0) + 1

        # Create a sorted list of unique damage values
        unique_damages = sorted(freq.keys())
        n = len(unique_damages)

        # Precompute total contribution for each unique damage
        total = [unique_damages[i] * freq[unique_damages[i]] for i in range(n)]

        # dp[i] will store the maximum possible damage when considering up to index i
        dp = [0] * n
        dp[0] = total[0]

        # Helper array to find the largest index j < i such that unique_damages[j] <= unique_damages[i] - 3
        # We can use binary search on unique_damages for (unique_damages[i] - 2)
        # because we need unique_damages[j] < unique_damages[i] - 2.
        for i in range(1, n):
            # Find via binary search the rightmost index j where unique_damages[j] < unique_damages[i] - 2
            # This is effectively looking for the insertion point of unique_damages[i] - 2 into the array
            # then subtract 1 to get the largest valid j.
            cur_val = unique_damages[i] - 2
            j = bisect_left(unique_damages, cur_val) - 1  # j might become -1 if no valid index is found

            if j >= 0:
                dp[i] = max(dp[i-1], dp[j] + total[i])
            else:
                # Means we can't pair with anything before i
                dp[i] = max(dp[i-1], total[i])

        return dp[-1]