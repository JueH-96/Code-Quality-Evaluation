from typing import List
from collections import Counter

class Solution:
    def findXSum(self, nums: List[int], k: int, x: int) -> List[int]:
        n = len(nums)
        res = []
        # Iterate over each sliding window of length k
        for i in range(n - k + 1):
            window = nums[i:i+k]
            freq = Counter(window)
            # If distinct elements are less than x, answer is sum(window)
            if len(freq) < x:
                res.append(sum(window))
            else:
                # Sort the items by frequency descending, tie break: value descending
                # sorted() sorts in ascending order, so we use reverse=True 
                sorted_items = sorted(freq.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
                # Only keep top x keys
                keep_keys = {num for num, _ in sorted_items[:x]}
                # Sum only occurrences of these top x most frequent numbers
                window_sum = sum(val for val in window if val in keep_keys)
                res.append(window_sum)
        return res

# For testing
if __name__ == "__main__":
    sol = Solution()
    # Example 1:
    nums1 = [1,1,2,2,3,4,2,3]
    k1 = 6
    x1 = 2
    print(sol.findXSum(nums1, k1, x1))  # Expected: [6, 10, 12]

    # Example 2:
    nums2 = [3,8,7,8,7,5]
    k2 = 2
    x2 = 2
    print(sol.findXSum(nums2, k2, x2))  # Expected: [11, 15, 15, 15, 12]