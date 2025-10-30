class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        n = len(nums)
        ans = 0
        target = bin(k)[2:]
        target = target.zfill(max(24 - len(target), 0) + len(target))
        cur = ''.join([bin(num)[2:].zfill(24) for num in nums])
        for i in range(len(target)):
            if target[i] == '0':
                continue
            idx = cur.find('1', i)
            if idx == -1:
                ans += 1
            else:
                cur = cur[:idx] + '0' + cur[idx + 1:]
        return ans