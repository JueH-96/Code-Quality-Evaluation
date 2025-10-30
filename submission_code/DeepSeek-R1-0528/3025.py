class Solution:
    def minOperations(self, nums: List[int], target: int) -> int:
        freq = [0] * 61
        for num in nums:
            exp = num.bit_length() - 1
            if exp < 61:
                freq[exp] += 1
        
        ops = 0
        for i in range(61):
            if target & (1 << i):
                if freq[i] > 0:
                    freq[i] -= 1
                else:
                    j = i + 1
                    while j < 61 and freq[j] == 0:
                        j += 1
                    if j == 61:
                        return -1
                    ops += (j - i)
                    freq[j] -= 1
                    for k in range(i, j):
                        freq[k] += 1
                    freq[i] -= 1
            if i < 60:
                freq[i+1] += freq[i] // 2
        return ops