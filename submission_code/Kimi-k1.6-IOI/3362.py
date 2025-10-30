class Solution:
    def medianOfUniquenessArray(self, nums: List[int]) -> int:
        n = len(nums)
        total_subarrays = n * (n + 1) // 2
        m = (total_subarrays - 1) // 2  # 0-based index of median
        
        low, high = 1, n
        answer = n  # Initialize with maximum possible value
        
        while low <= high:
            mid = (low + high) // 2
            cnt = self.count_at_most(nums, mid)
            if cnt >= (m + 1):
                answer = mid
                high = mid - 1
            else:
                low = mid + 1
        
        return answer
    
    def count_at_most(self, nums: List[int], x: int) -> int:
        max_num = 10**5 + 2  # Since nums[i] <= 1e5
        freq = [0] * max_num
        left = 0
        distinct = 0
        count = 0
        
        for right in range(len(nums)):
            num = nums[right]
            if freq[num] == 0:
                distinct += 1
            freq[num] += 1
            
            # Shrink the window from the left if distinct exceeds x
            while distinct > x:
                left_num = nums[left]
                freq[left_num] -= 1
                if freq[left_num] == 0:
                    distinct -= 1
                left += 1
            
            # Add the number of valid subarrays ending at right
            count += right - left + 1
        
        return count