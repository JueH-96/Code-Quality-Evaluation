class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        # The smallest element removal process effectively forces us
        # to remove all elements that are strictly less than k in order
        # to ensure that every element in the array is at least k.
        # Thus, the answer is simply the count of elements < k.
        
        return sum(1 for x in nums if x < k)