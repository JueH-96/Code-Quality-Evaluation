class Solution:
    def numberGame(self, nums: List[int]) -> List[int]:
        nums.sort()
        arr = []
        
        for i in range(0, len(nums), 2):
            # Bob appends first, then Alice
            arr.append(nums[i+1])  # Bob's element (larger of the pair)
            arr.append(nums[i])    # Alice's element (smaller of the pair)
        
        return arr