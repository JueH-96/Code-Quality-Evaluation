class Solution:
    def numberGame(self, nums: List[int]) -> List[int]:
        arr = []
        while nums:
            # Alice removes the minimum element
            min_value = min(nums)
            nums.remove(min_value)
            alice_value = min_value
            
            # Bob removes the next minimum element
            min_value = min(nums)
            nums.remove(min_value)
            bob_value = min_value
            
            # Bob appends first, then Alice
            arr.append(bob_value)
            arr.append(alice_value)
        
        return arr