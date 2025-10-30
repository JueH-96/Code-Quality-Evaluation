class Solution:
    def numberGame(self, nums: List[int]) -> List[int]:
        nums.sort()
        arr = []
        for i in range(0, len(nums), 2):
            # Alice removes the nums[i] (the current minimum),
            # Bob removes the nums[i+1] (the next minimum),
            # Bob appends first, then Alice.
            arr.append(nums[i+1])
            arr.append(nums[i])
        return arr