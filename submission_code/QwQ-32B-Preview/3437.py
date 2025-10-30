from typing import List
from collections import Counter, deque

class Solution:
    def maximumTotalDamage(self, power: List[int]) -> int:
        # Count frequency of each damage value
        freq = Counter(power)
        # Sort unique damage values
        s = sorted(freq.keys())
        m = len(s)
        if m == 0:
            return 0
        # Initialize dp array
        dp = [0] * m
        # Initialize deque to keep track of j where s[j] <= s[i] -3
        window = deque()
        # Initialize dp[0]
        if s[0] -3 <= 0:
            dp[0] = freq[s[0]] * s[0]
            window.append(0)
        else:
            dp[0] = freq[s[0]] * s[0]
        # Populate dp array
        for i in range(1, m):
            # Find dp[j] where s[j] <= s[i] -3
            while window and s[window[-1]] > s[i] -3:
                window.pop()
            if window:
                dp[i] = max(dp[i-1], dp[window[-1]] + freq[s[i]] * s[i])
            else:
                dp[i] = max(dp[i-1], freq[s[i]] * s[i])
            # Maintain the window
            while window and dp[i] >= dp[window[0]]:
                window.popleft()
            window.appendleft(i)
        return dp[m-1]