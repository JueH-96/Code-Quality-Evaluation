class Solution:
    def minOperations(self, nums: List[int], target: int) -> int:
        from collections import defaultdict
        import math

        freq = defaultdict(int)
        for num in nums:
            freq[num] +=1

        operations =0
        max_power = max(freq.keys()) if freq else 0

        for i in range(32):
            power = 1 <<i
            if target & power:
                if freq[power] >0:
                    freq[power] -=1
                else:
                    j = i+1
                    while j <32 and freq[1<<j]==0:
                        j +=1
                    if j ==32:
                        return -1
                    for k in range(j, i, -1):
                        freq[1<<k] -=1
                        freq[1<<(k-1)] +=2
                        operations +=1
                    freq[power] -=1
            freq[power+1] += freq[power]//2

        return operations