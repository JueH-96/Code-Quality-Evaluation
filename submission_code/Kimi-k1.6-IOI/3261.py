class Solution:
    def minOrAfterOperations(self, nums: List[int], k: int) -> int:
        n = len(nums)
        m = n - k
        ans = 0
        
        for b in range(29, -1, -1):
            higher_mask = (-1) << (b + 1)
            required_higher = ans & higher_mask
            bit_b_mask = 1 << b
            current = -1
            partitions = 0
            valid = True
            
            for num in nums:
                if current == -1:
                    current = num
                    if ((current & higher_mask) != required_higher) or ((current & bit_b_mask) != 0):
                        valid = False
                        break
                else:
                    temp = current & num
                    if ((temp & higher_mask) == required_higher) and ((temp & bit_b_mask) == 0):
                        current = temp
                    else:
                        partitions += 1
                        current = num
                        if ((current & higher_mask) != required_higher) or ((current & bit_b_mask) != 0):
                            valid = False
                            break
            
            if current != -1:
                partitions += 1
            
            if valid and partitions <= m:
                ans &= ~ (1 << b)
            else:
                ans |= (1 << b)
        
        return ans