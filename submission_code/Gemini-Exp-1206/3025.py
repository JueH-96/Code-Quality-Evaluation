class Solution:
    def minOperations(self, nums: List[int], target: int) -> int:
        count = [0] * 31
        for num in nums:
            count[num.bit_length() - 1] += 1
        if sum(nums) < target:
            return -1
        ans = 0
        for i in range(30):
            if (target >> i) & 1:
                if count[i] > 0:
                    count[i] -= 1
                else:
                    j = i
                    while j < 30 and count[j] == 0:
                        j += 1
                        ans += 1
                    count[j] -= 1
                    for k in range(i, j):
                        count[k] += 1
            count[i + 1] += count[i] // 2
        return ans