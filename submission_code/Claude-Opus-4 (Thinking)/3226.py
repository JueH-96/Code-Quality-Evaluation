class Solution:
    def numberGame(self, nums: List[int]) -> List[int]:
        nums.sort()
        result = []
        
        # Process pairs of elements
        for i in range(0, len(nums), 2):
            # Alice takes nums[i] (smaller), Bob takes nums[i+1] (larger)
            # But Bob appends first, then Alice
            result.append(nums[i+1])
            result.append(nums[i])
        
        return result