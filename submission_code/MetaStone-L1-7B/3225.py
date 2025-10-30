class Solution:
    def maxSubarrayLength(self, nums: List[int], k: int) -> int:
        count = defaultdict(int)
        left = 0
        max_len = 0
        violations = 0
        
        for right in range(len(nums)):
            num = nums[right]
            count[num] += 1
            if count[num] == k + 1:
                violations += 1
            
            while violations > 0:
                left_num = nums[left]
                count[left_num] -= 1
                if count[left_num] == k:
                    violations -= 1
                left += 1
            
            current_len = right - left + 1
            if current_len > max_len:
                max_len = current_len
        
        return max_len