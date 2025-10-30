class Solution:
    def minOperations(self, nums: List[int]) -> int:
        count = Counter(nums)
        operations = 0
        for num in count:
            if count[num] == 1:
                return -1
            elif count[num] == 2:
                operations += 1
            elif count[num] == 3:
                operations += 1
            elif count[num] == 4:
                operations += 2
            elif count[num] == 5:
                operations += 2
            elif count[num] % 3 == 0:
                operations += count[num] // 3
            elif count[num] % 3 == 1:
                operations += count[num] // 3 + 1
            elif count[num] % 3 == 2:
                operations += count[num] // 3 + 1
        return operations