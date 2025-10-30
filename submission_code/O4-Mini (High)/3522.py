from typing import List

class Solution:
    def resultsArray(self, nums: List[int], k: int) -> List[int]:
        n = len(nums)
        result = []
        # Slide over each window of size k
        for i in range(n - k + 1):
            ok = True
            # Check if the window is strictly increasing by 1 each step
            for j in range(i, i + k - 1):
                if nums[j + 1] != nums[j] + 1:
                    ok = False
                    break
            # If it is, the power is the last (and largest) element; otherwise -1
            result.append(nums[i + k - 1] if ok else -1)
        return result