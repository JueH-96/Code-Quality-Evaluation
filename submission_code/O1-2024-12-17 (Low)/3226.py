class Solution:
    def numberGame(self, nums: List[int]) -> List[int]:
        arr = []
        
        # While there are still elements in nums
        while nums:
            # Alice removes the smallest element
            alice_min = min(nums)
            nums.remove(alice_min)
            
            # Bob removes the smallest element
            bob_min = min(nums)
            nums.remove(bob_min)
            
            # Bob appends first, then Alice
            arr.append(bob_min)
            arr.append(alice_min)
        
        return arr