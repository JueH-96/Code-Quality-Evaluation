class Solution:
    def maximumTotalDamage(self, power: List[int]) -> int:
        from collections import Counter
        import bisect
        
        # Count how many times each damage value occurs
        freq_map = Counter(power)
        
        # For each distinct damage d, the total contribution if we choose damage d
        # is d * (number of spells with damage d).
        freq = {}
        for d, count in freq_map.items():
            freq[d] = d * count
        
        # Sort the distinct damage values
        distinct_damages = sorted(freq.keys())
        n = len(distinct_damages)
        
        # nextIndex[i] will hold the smallest index j > i such that
        # distinct_damages[j] >= distinct_damages[i] + 3
        nextIndex = [0] * n
        for i in range(n):
            d = distinct_damages[i]
            # Find where d+3 would be inserted to keep the array sorted
            j = bisect.bisect_left(distinct_damages, d + 3)
            nextIndex[i] = j
        
        # dp[i] = the maximum total damage using distinct_damages[i..]
        dp = [0] * (n + 1)
        
        # Fill dp from the end moving backwards
        for i in range(n - 1, -1, -1):
            dp[i] = max(
                dp[i + 1],  # Option 1: skip this damage value
                freq[distinct_damages[i]] + dp[nextIndex[i]]  # Option 2: pick this damage
            )
        
        return dp[0]