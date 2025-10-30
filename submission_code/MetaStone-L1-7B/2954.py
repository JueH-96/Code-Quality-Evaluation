class Solution:
    def maxSum(self, nums: List[int], m: int, k: int) -> int:
        n = len(nums)
        if n == 0:
            return 0
        
        current_sum = sum(nums[:k])
        current_set = set(nums[:k])
        max_sum = current_sum if len(current_set) >= m else 0
        
        for i in range(k, n):
            outgoing = nums[i - k]
            incoming = nums[i]
            
            current_sum += incoming - outgoing
            
            if outgoing in current_set:
                current_set.remove(outgoing)
            
            if incoming not in current_set:
                current_set.add(incoming)
            
            if len(current_set) >= m:
                if current_sum > max_sum:
                    max_sum = current_sum
        
        return max_sum if max_sum != 0 else 0