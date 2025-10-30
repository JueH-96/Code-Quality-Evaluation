class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        result = 0
        prev = {}  # AND value -> count of subarrays ending at previous position
        
        for num in nums:
            curr = {}
            # Subarray of just this element
            curr[num] = curr.get(num, 0) + 1
            
            # Extend previous subarrays
            for and_val, count in prev.items():
                new_and = and_val & num
                curr[new_and] = curr.get(new_and, 0) + count
            
            # Count subarrays with AND equal to k
            if k in curr:
                result += curr[k]
            
            prev = curr
        
        return result