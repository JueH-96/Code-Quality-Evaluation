class Solution:
    def numberGame(self, nums: List[int]) -> List[int]:
        nums.sort()
        arr = []
        
        for i in range(0, len(nums), 2):
            # Bob appends Alice's removed number
            arr.append(nums[i+1])
            # Alice appends Bob's removed number
            arr.append(nums[i])
            
        return arr