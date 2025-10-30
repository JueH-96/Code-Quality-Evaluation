class Solution:
    def maxGoodNumber(self, nums: List[int]) -> int:
        def get_binary(n):
            return bin(n)[2:]
            
        def concat_to_decimal(a, b, c):
            binary = get_binary(a) + get_binary(b) + get_binary(c)
            return int(binary, 2)
            
        max_num = 0
        # Try all possible permutations
        for a in nums:
            for b in nums:
                for c in nums:
                    if len({(a,b,c)}) == 1 and nums.count(a) < 3:
                        continue
                    if len({(a,b,c)}) == 2 and nums.count(a) < 2 and a in (b,c):
                        continue
                    max_num = max(max_num, concat_to_decimal(a, b, c))
                        
        return max_num