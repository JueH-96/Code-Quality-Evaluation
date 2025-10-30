class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        count = [0, 0, 0]
        for num in nums:
            count[num % 3] += 1
        ans = 0
        if count[1] == count[2]:
            ans = count[1]
        else:
            diff = abs(count[1] - count[2])
            ans += diff
            count[1] -= diff // 3 * 1
            count[2] -= diff // 3 * 2
            count[0] -= diff // 3 * 0
            
            ans += count[1] + count[2]
        return ans