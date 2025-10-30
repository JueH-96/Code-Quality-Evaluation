from typing import List

class Solution:
    def numberGame(self, nums: List[int]) -> List[int]:
        # Sort nums so that each round we can take the two smallest easily
        nums.sort()
        arr = []
        # Process in pairs: nums[0], nums[1] are smallest two, then nums[2], nums[3], etc.
        for i in range(0, len(nums), 2):
            # Alice removes nums[i], Bob removes nums[i+1]
            # Bob appends first, then Alice
            arr.append(nums[i + 1])
            arr.append(nums[i])
        return arr