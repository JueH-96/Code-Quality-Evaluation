from typing import List

class Solution:
    def medianOfUniquenessArray(self, nums: List[int]) -> int:
        n = len(nums)
        total_sub = n * (n + 1) // 2
        # The median is defined as the element at the 1-indexed position:
        #   ceil(total_sub / 2)
        # In our binary search we use target = (total_sub + 1) // 2.
        target = (total_sub + 1) // 2
        
        # Maximum distinct count in any subarray equals the distinct count of the whole array.
        D = len(set(nums))
        
        def at_most(k: int) -> int:
            """
            Returns the number of subarrays with at most k distinct elements.
            Uses a standard two-pointer (sliding window) approach.
            """
            count = 0
            left = 0
            freq = {}
            distinct = 0
            for right, num in enumerate(nums):
                freq[num] = freq.get(num, 0) + 1
                if freq[num] == 1:
                    distinct += 1
                # Shrink the window if distinct count exceeds k
                while distinct > k:
                    freq[nums[left]] -= 1
                    if freq[nums[left]] == 0:
                        distinct -= 1
                    left += 1
                # All subarrays ending at 'right' with start in [left, right] are valid.
                count += (right - left + 1)
            return count
        
        # Binary search for the smallest k in [1, D] such that the number of subarrays 
        # with at most k distinct values is at least the median position.
        low, high = 1, D
        ans = D  # Default answer; it will get updated.
        while low <= high:
            mid = (low + high) // 2
            if at_most(mid) >= target:
                ans = mid
                high = mid - 1
            else:
                low = mid + 1
                
        return ans

# You can run some tests to verify the solution:
if __name__ == "__main__":
    sol = Solution()
    # Example 1:
    nums1 = [1, 2, 3]
    print(sol.medianOfUniquenessArray(nums1))  # Expected output: 1
    
    # Example 2:
    nums2 = [3, 4, 3, 4, 5]
    print(sol.medianOfUniquenessArray(nums2))  # Expected output: 2
    
    # Example 3:
    nums3 = [4, 3, 5, 4]
    print(sol.medianOfUniquenessArray(nums3))  # Expected output: 2