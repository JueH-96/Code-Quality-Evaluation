class Solution:
    def numberGame(self, nums: List[int]) -> List[int]:
        arr = []
        nums.sort()  # Sort the nums array to easily access the minimum elements
        while nums:
            # Alice removes the minimum element
            alice_min = nums.pop(0)
            # Bob removes the next minimum element
            bob_min = nums.pop(0)
            # Bob appends his removed element first
            arr.append(bob_min)
            # Alice appends her removed element
            arr.append(alice_min)
        return arr