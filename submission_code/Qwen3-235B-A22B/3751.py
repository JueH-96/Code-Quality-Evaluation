class Solution:
    def maxFrequency(self, nums: List[int], k: int) -> int:
        original_k_count = nums.count(k)
        possible_x = {k - a for a in nums}
        max_gain = float('-inf')
        
        for x in possible_x:
            current_sum = 0
            best_sum = float('-inf')
            
            for a in nums:
                delta_val = ((a + x == k) - (a == k))
                current_sum = max(delta_val, current_sum + delta_val)
                if current_sum > best_sum:
                    best_sum = current_sum
            
            if best_sum > max_gain:
                max_gain = best_sum
        
        return original_k_count + (max_gain if max_gain != float('-inf') else 0)