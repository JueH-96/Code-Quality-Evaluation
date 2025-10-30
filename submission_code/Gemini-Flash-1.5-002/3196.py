class Solution:
    def maxFrequencyScore(self, nums: List[int], k: int) -> int:
        count = {}
        for num in nums:
            count[num] = count.get(num, 0) + 1
        
        sorted_keys = sorted(count.keys())
        
        prefix_sum = {}
        current_sum = 0
        for key in sorted_keys:
            current_sum += count[key] * key
            prefix_sum[key] = current_sum
        
        max_freq = 0
        
        for i in range(len(sorted_keys)):
            target = sorted_keys[i]
            
            left = 0
            right = i
            
            while left <= right:
                mid = (left + right) // 2
                
                cost = 0
                if mid > 0:
                    cost = (target * (count[sorted_keys[mid]])) - (prefix_sum[sorted_keys[mid-1]] + count[sorted_keys[mid]] * sorted_keys[mid])
                else:
                    cost = (target * (count[sorted_keys[mid]])) - prefix_sum[sorted_keys[mid]]
                    
                if cost <= k:
                    left = mid + 1
                else:
                    right = mid -1
            
            
            freq = 0
            for j in range(right+1):
                freq += count[sorted_keys[j]]
            
            max_freq = max(max_freq, freq)
            
        return max_freq