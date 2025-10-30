class Solution:
    def minOperations(self, nums: List[int], target: int) -> int:
        counts = Counter(nums)
        operations = 0
        for i in range(31):
            power_of_2 = 1 << i
            if target & power_of_2:
                if counts[power_of_2] > 0:
                    counts[power_of_2] -= 1
                else:
                    found = False
                    for j in range(i + 1, 32):
                        power = 1 << j
                        if counts[power] > 0:
                            counts[power] -= 1
                            for k in range(i, j):
                                counts[1 << k] += 1
                            operations += j - i
                            found = True
                            break
                    if not found:
                        return -1
        return operations