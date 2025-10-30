class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        # If the original list is already distinct, return 0.
        if len(nums) == len(set(nums)):
            return 0
        
        # Starting from idx representing operations count times 3 removal from beginning.
        n = len(nums)
        # 0 operations means full array.
        # With each operation, we remove the first 3 elements.
        # We try from 0 upto the number of operations needed and check if the remaining elements are distinct.
        operations = 0
        while operations * 3 < n:
            # Compute remaining part after operations operations.
            remain = nums[operations * 3:]
            if len(remain) == len(set(remain)):
                return operations
            operations += 1
        
        # If we are here, then even after all possible removals (resulting in an empty array which is distinct)
        # we return operations. Here operations * 3 would be >= n, which means array becomes empty.
        return operations