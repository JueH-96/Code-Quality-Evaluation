class Solution:
    def maxFrequencyScore(self, nums: List[int], k: int) -> int:
        nums.sort()
        n = len(nums)
        if n == 0:
            return 0
        
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i+1] = prefix[i] + nums[i]
        
        max_freq = 1
        for r in range(n):
            low = 0
            high = r
            best_l = r  # Initialize to the worst case
            
            while low <= high:
                mid = (low + high) // 2
                m = mid + (r - mid) // 2
                m = min(m, r)  # Ensure m doesn't exceed r
                
                # Calculate sum_l_to_m
                sum_l_to_m = prefix[m+1] - prefix[mid]
                # Calculate sum_m_plus_1_to_r
                sum_m_plus_1_to_r = prefix[r+1] - prefix[m+1]
                
                # Compute cost
                cost = (m - mid + 1) * nums[m] - sum_l_to_m + sum_m_plus_1_to_r - (r - m) * nums[m]
                
                if cost <= k:
                    best_l = mid
                    high = mid - 1
                else:
                    low = mid + 1
            
            if best_l != r:
                current_max = r - best_l + 1
                if current_max > max_freq:
                    max_freq = current_max
        
        return max_freq