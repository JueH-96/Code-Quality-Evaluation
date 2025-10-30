class Solution:
    def maxGoodNumber(self, nums: List[int]) -> int:
        def to_binary(num):
            return bin(num)[2:]
        
        binary_nums = [to_binary(num) for num in nums]
        
        max_num = 0
        for perm in itertools.permutations(binary_nums):
            concatenated = ''.join(perm)
            max_num = max(max_num, int(concatenated, 2))
        
        return max_num