class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        count = 0
        prev_ands = {}  # and_value -> count of subarrays ending at previous position
        
        for num in nums:
            curr_ands = {}
            
            # Single element subarray ending at current position
            curr_ands[num] = curr_ands.get(num, 0) + 1
            
            # Extend all subarrays ending at previous position
            for and_val, cnt in prev_ands.items():
                new_and = and_val & num
                curr_ands[new_and] = curr_ands.get(new_and, 0) + cnt
            
            # Count subarrays ending at current position with AND equal to k
            count += curr_ands.get(k, 0)
            
            # Update for next iteration
            prev_ands = curr_ands
        
        return count