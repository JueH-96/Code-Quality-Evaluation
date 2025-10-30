class Solution:
    def numberGame(self, nums: List[int]) -> List[int]:
        # Sort the array to easily get minimum elements
        nums.sort()
        arr = []
        
        # Process pairs of elements
        for i in range(0, len(nums), 2):
            alice_element = nums[i]      # Alice gets the smaller element
            bob_element = nums[i + 1]    # Bob gets the next smaller element
            
            # Bob appends first, then Alice
            arr.append(bob_element)
            arr.append(alice_element)
        
        return arr