class Solution:
    def minimumDifference(self, nums: List[int], k: int) -> int:
        result = infie = 1e20
        account, low, big = [0] * 32, 0, 0
        
        for i, x in enumerate(nums):
            for j in range(32):
                kth_bit = x & (1 << j)  # check the kth bit of element
                if kth_bit:
                    account[j] += 1  # record the element with kth bit be 1
            
            if all(a == i + 1 - low for a in account):
                result = min(result, abs((big | (1 << 32)) - k))
                result = min(result, abs((low - 1) - k))
            
            while i - big >= 30:
                if not ((1 << (i - big)) & big):
                    big |= (1 << i - big)
            
            while i - low >= 30:
                if low and (1 << (i - low)) & low:
                    for j in range(32):
                        if low & (1 << j):
                            account[j] -= 1
                    low = low & (low - 1)
        return result