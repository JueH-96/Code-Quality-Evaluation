class Solution:
    def minOperations(self, nums: List[int], target: int) -> int:
        count = [0] * 31
        for num in nums:
            count[num.bit_length() - 1] += 1
        
        operations = 0
        carry = 0
        
        for i in range(31):
            if target & (1 << i):
                if count[i]:
                    count[i] -= 1
                elif carry:
                    carry -= 1
                else:
                    j = i
                    while j < 31 and count[j] == 0:
                        j += 1
                    if j == 31:
                        return -1
                    operations += j - i
                    count[j] -= 1
                    for k in range(i, j):
                        count[k] += 1
                    carry += 1
            
            carry += count[i]
            count[i+1] += carry // 2
            carry %= 2
        
        return operations