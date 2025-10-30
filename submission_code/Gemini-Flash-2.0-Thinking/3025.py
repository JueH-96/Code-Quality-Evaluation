class Solution:
    def minOperations(self, nums: List[int], target: int) -> int:
        counts = {}
        for num in nums:
            counts[num] = counts.get(num, 0) + 1

        operations = 0
        for i in range(31):
            if (target >> i) & 1:
                power_of_2 = 1 << i
                if counts.get(power_of_2, 0) > 0:
                    counts[power_of_2] -= 1
                else:
                    found = False
                    for j in range(i + 1, 31):
                        higher_power_of_2 = 1 << j
                        if counts.get(higher_power_of_2, 0) > 0:
                            operations += (j - i)
                            counts[higher_power_of_2] -= 1
                            for k in range(i, j):
                                intermediate_power = 1 << k
                                counts[intermediate_power] = counts.get(intermediate_power, 0) + 1
                            found = True
                            break
                    if not found:
                        return -1

        return operations