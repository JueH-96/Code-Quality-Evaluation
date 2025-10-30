from collections import defaultdict

class Solution:
    def sumImbalanceNumbers(self, nums: List[int]) -> int:
        n = len(nums)
        sum_imbalance = 0
        
        for start in range(n):
            min_val = float('inf')
            max_val = float('-inf')
            last_seen = defaultdict(int)
            imbalance_count = 0
            
            for end in range(start, n):
                num = nums[end]
                min_val = min(min_val, num)
                max_val = max(max_val, num)
                
                if num not in last_seen:
                    if min_val != max_val:
                        imbalance_count += 1
                
                sum_imbalance += imbalance_count
                
                last_seen[num] = end
                
        return sum_imbalance