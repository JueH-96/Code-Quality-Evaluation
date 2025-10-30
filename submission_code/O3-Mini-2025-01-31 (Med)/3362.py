from collections import defaultdict
from typing import List

class Solution:
    def medianOfUniquenessArray(self, nums: List[int]) -> int:
        # Helper function: returns number of subarrays with at most k distinct numbers.
        def atMost(k: int) -> int:
            # Two pointers technique.
            count = 0
            left = 0
            freq = defaultdict(int)
            distinct = 0
            for right, val in enumerate(nums):
                # include current element
                if freq[val] == 0:
                    distinct += 1
                freq[val] += 1
                # shrink window until it has at most k distinct numbers
                while distinct > k:
                    freq[nums[left]] -= 1
                    if freq[nums[left]] == 0:
                        distinct -= 1
                    left += 1
                # all windows ending at right with start in [left, right] are valid
                count += (right - left + 1)
            return count

        n = len(nums)
        # Total number of subarrays.
        total_subarrays = n * (n + 1) // 2
        # In the sorted uniqueness array (which is a multiset of distinct counts per subarray),
        # the median is at index (total_subarrays - 1) // 2 (0-indexed). 
        median_index = (total_subarrays - 1) // 2

        # Let F(x) be the number of subarrays with at most x distinct numbers.
        # Notice that each subarray has a distinct count d, and subarrays with exactly d distinct numbers
        # contribute (F(d) - F(d-1)) to the overall count of subarrays.
        # The uniqueness array (when sorted) has F(1) copies of 1, then F(2)-F(1) copies of 2, etc.
        # So, the median is the smallest d such that F(d) > median_index.
        global_distinct = len(set(nums))
        
        low, high = 1, global_distinct
        answer = high
        while low <= high:
            mid = (low + high) // 2
            count_at_most = atMost(mid)
            if count_at_most > median_index:
                answer = mid
                high = mid - 1
            else:
                low = mid + 1
        return answer

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    print(sol.medianOfUniquenessArray([1,2,3]))      # Output: 1
    print(sol.medianOfUniquenessArray([3,4,3,4,5]))    # Output: 2
    print(sol.medianOfUniquenessArray([4,3,5,4]))      # Output: 2