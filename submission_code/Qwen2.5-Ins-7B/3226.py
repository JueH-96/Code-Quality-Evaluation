class Solution:
    def numberGame(self, nums: List[int]) -> List[int]:
        nums.sort()
        arr = []
        while nums:
            arr.append(nums.pop())  # Bob removes the minimum element and appends it to arr
            arr.append(nums.pop())  # Alice removes the new minimum element and appends it to arr
        return arr