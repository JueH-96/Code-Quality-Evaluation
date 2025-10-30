class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        nums.sort()
        count = 0
        while nums and nums[-1] >= k:
            num = nums.pop()
            if num == k:
                return count
            elif num < 2 * k:
                k -= num
                count += 1
            else:
                nums.append(num - k)
                count += 1
        return count