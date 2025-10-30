class Solution:
    def numberOfSequence(self, n: int, sick: List[int]) -> int:
        M = 10 ** 9 + 7
        len_sick = len(sick)
        sick.append(n)
        
        # min_value, center, max_value
        # https://i.postimg.cc/j2RRb0R4/disease-queue-diagram.png
        max_len = len_sick + 1
        nums = [sick[0] + 1, *[sick[i] - sick[i - 1] - 1 for i in range(1, len_sick)], n - sick[len_sick - 1]]
        result = 1
        
        # 35*4 = 140   ==   (1 << 10) * 14
        # power 2^35 = (1<<35)
        pre = [1 << temp - 1 if temp != 0 else 1 for temp in nums]
        last = [1 << temp if temp != 0 else 1 for temp in nums]

        for i in range(max_len):
            if nums[i] < 2:
                continue
            result = result * pre[i] * last[i] % M
            
            if 0 < i < max_len - 1 and nums[i - 1] > 0 and nums[i + 1] > 0:
                result = result * fast(pow, M, nums[i], nums[i - 1]) % M * fast(pow, M, nums[i], nums[i + 1]) % M
            
            elif nums[i - 1] > 0 and nums[i + 1] == 0:
                result = result * fast(pow, M, nums[i], nums[i - 1]) % M
            
            elif nums[i - 1] == 0 and nums[i + 1] > 0:
                result = result * fast(pow, M, nums[i], nums[i + 1]) % M
        
        return result % M

def fast(pow, m, x, n):
    ans = 1
    x %= m

    while n > 0:
        if n & 1:
            ans *= x % m
            ans %= m
        x = x * x % m
        n >>= 1
        
    return ans