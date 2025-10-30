class Solution:
    def numberGame(self, nums: List[int]) -> List[int]:
        nums.sort()  # Sort the array in ascending order
        arr = []
        
        for i in range(0, len(nums), 2):
            # Bob appends Alice's removed element
            arr.append(nums[i+1])
            # Alice appends Bob's removed element
            arr.append(nums[i])
        
        return arr