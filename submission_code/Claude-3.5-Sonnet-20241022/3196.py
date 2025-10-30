class Solution:
    def maxFrequencyScore(self, nums: List[int], k: int) -> int:
        n = len(nums)
        nums.sort()
        
        def canAchieveFreq(freq):
            if freq > n:
                return False
                
            # Try each possible window of size freq
            for i in range(n - freq + 1):
                window = nums[i:i+freq]
                target = window[freq//2]  # Choose middle element as target
                
                # Calculate total operations needed
                ops = 0
                for num in window:
                    ops += abs(num - target)
                
                if ops <= k:
                    return True
                    
            return False
            
        # Binary search on the frequency
        left, right = 1, n
        ans = 1
        
        while left <= right:
            mid = (left + right) // 2
            
            if canAchieveFreq(mid):
                ans = mid
                left = mid + 1
            else:
                right = mid - 1
                
        return ans