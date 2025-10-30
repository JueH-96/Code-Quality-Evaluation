class Solution:
    def minLength(self, s: str, numOps: int) -> int:
        n = len(s)
        left = 1
        right = n
        answer = n  # Initialize with the maximum possible value
        
        while left <= right:
            mid = (left + right) // 2
            if self.is_possible(s, numOps, mid):
                answer = mid
                right = mid - 1
            else:
                left = mid + 1
        
        return answer
    
    def is_possible(self, s: str, numOps: int, k: int) -> bool:
        n = len(s)
        if k == 0:
            return False  # No valid run length possible
        
        # Initialize DP table for the first character
        prev_dp = [[float('inf')] * (k + 1) for _ in range(2)]
        first_char = int(s[0])
        prev_dp[first_char][1] = 0
        prev_dp[1 - first_char][1] = 1
        
        for i in range(1, n):
            current_char = int(s[i])
            current_dp = [[float('inf')] * (k + 1) for _ in range(2)]
            
            for prev_char in [0, 1]:
                for prev_len in range(1, k + 1):
                    if prev_dp[prev_char][prev_len] == float('inf'):
                        continue
                    
                    # Option 1: Do not flip the current character
                    new_char = current_char
                    cost = prev_dp[prev_char][prev_len]
                    if new_char == prev_char:
                        new_len = prev_len + 1
                        if new_len <= k:
                            if current_dp[new_char][new_len] > cost:
                                current_dp[new_char][new_len] = cost
                    else:
                        new_len = 1
                        if current_dp[new_char][new_len] > cost:
                            current_dp[new_char][new_len] = cost
                    
                    # Option 2: Flip the current character
                    new_char = 1 - current_char
                    cost = prev_dp[prev_char][prev_len] + 1
                    if new_char == prev_char:
                        new_len = prev_len + 1
                        if new_len <= k:
                            if current_dp[new_char][new_len] > cost:
                                current_dp[new_char][new_len] = cost
                    else:
                        new_len = 1
                        if current_dp[new_char][new_len] > cost:
                            current_dp[new_char][new_len] = cost
            
            prev_dp = current_dp
        
        # Find the minimum flips required among all possible states
        min_flips = float('inf')
        for c in [0, 1]:
            for l in range(1, k + 1):
                if prev_dp[c][l] < min_flips:
                    min_flips = prev_dp[c][l]
        
        return min_flips <= numOps