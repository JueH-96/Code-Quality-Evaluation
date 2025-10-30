class Solution:
    def maxScore(self, points: List[int], m: int) -> int:
        n = len(points)
        if m >= 2 * n - 1:
            # If we have enough moves to cover all positions back and forth
            return min(points)
        
        # We use a sliding window technique to find the maximum possible minimum
        # value of gameScore that can be achieved with exactly `m` moves.
        # This involves considering all possible distributions of `m` moves.
        
        # Prefix sums to calculate the sum of points in constant time
        prefix_sum = [0] * (n + 1)
        for i in range(n):
            prefix_sum[i + 1] = prefix_sum[i] + points[i]
        
        def sum_range(l, r):
            return prefix_sum[r + 1] - prefix_sum[l]
        
        # We need to maximize the minimum score in the gameScore array
        # after exactly `m` moves.
        # Binary search on the answer for the maximum minimum value.
        low, high = 0, min(points)
        while low < high:
            mid = (low + high + 1) // 2
            # Check if we can achieve at least `mid` in all positions with `m` moves
            possible = False
            # dp[i] will be the minimum moves needed to make all first i elements >= mid
            dp = [float('inf')] * (n + 1)
            dp[0] = 0  # No moves needed to satisfy 0 elements
            
            # We use a deque to maintain the minimum dp values over a sliding window
            from collections import deque
            window = deque()
            
            for i in range(n):
                # We need points[i] >= mid, calculate how many times we need to visit this point
                if points[i] >= mid:
                    dp[i + 1] = dp[i]  # We can just move right without extra cost
                else:
                    # We need to accumulate enough visits to make gameScore[i] >= mid
                    required_visits = (mid + points[i] - 1) // points[i]  # ceil(mid / points[i])
                    if i + 1 >= required_visits:
                        # We can only consider this if we have enough moves to accumulate
                        while window and window[0] < i + 1 - required_visits:
                            window.popleft()
                        if window:
                            dp[i + 1] = min(dp[i + 1], dp[window[0]] + required_visits)
                
                # Maintain the deque for the sliding window minimum
                while window and dp[window[-1]] >= dp[i]:
                    window.pop()
                window.append(i)
            
            if dp[n] <= m:
                possible = True
            
            if possible:
                low = mid
            else:
                high = mid - 1
        
        return low