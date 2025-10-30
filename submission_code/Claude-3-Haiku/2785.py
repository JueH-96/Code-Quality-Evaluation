class Solution:
    def semiOrderedPermutation(self, nums: List[int]) -> int:
        n = len(nums)
        
        # Find the indices of 1 and n
        i1 = nums.index(1)
        in_ = nums.index(n)
        
        # If the permutation is already semi-ordered, return 0
        if i1 == 0 and in_ == n-1:
            return 0
        
        # Calculate the minimum number of operations required
        operations = 0
        
        # If 1 is not at the beginning, move it to the beginning
        if i1 > 0:
            operations += i1
            nums = nums[i1:] + nums[:i1]
            in_ = in_ - i1
        
        # If n is not at the end, move it to the end
        if in_ < n-1:
            operations += n - 1 - in_
            nums = nums[in_+1:] + nums[:in_+1]
        
        return operations