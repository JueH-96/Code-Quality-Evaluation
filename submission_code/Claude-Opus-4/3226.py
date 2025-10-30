class Solution:
    def numberGame(self, nums: List[int]) -> List[int]:
        # Sort the array to easily get minimum elements
        nums.sort()
        
        # Result array
        arr = []
        
        # Process pairs of elements
        for i in range(0, len(nums), 2):
            # Alice takes nums[i], Bob takes nums[i+1]
            # But Bob appends first, then Alice
            arr.append(nums[i+1])  # Bob's element
            arr.append(nums[i])    # Alice's element
        
        return arr