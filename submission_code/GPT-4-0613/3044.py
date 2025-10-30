class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        nums.reverse()
        collection = set()
        operations = 0
        for num in nums:
            collection.add(num)
            operations += 1
            if len(collection) == k:
                break
        return operations