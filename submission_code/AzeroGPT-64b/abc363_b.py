from typing import List

class HairGrowth:
    def daysToGrowHair(self, lengths: List[int], target: int, required_count: int) -> int:
        """
        Calculates the number of days after which the number of people whose hair length is at least T becomes P or more.
        
        :param lengths: List of current hair lengths.
        :param target: Target hair length.
        :param required_count: Required number of people with hair length at least T.
        :return: Number of days needed or 0 if already satisfied.
        """
        lengths = [max(0, target - length) for length in lengths]
        lengths.sort()
        
        if len(lengths) - lengths.count(0) < required_count - lengths[:required_count].count(0):
            return -1  # This condition cannot happen based on the problem statement
        
        i = required_count - lengths[:required_count].count(0) - 1
        if i > -1 and lengths[i] == lengths[i + 1]:
            i -= 1
        
        if i < 0:
            return 0
        else:
            return lengths[i]
        
# Read input
n, t, p = map(int, input().split())
l = list(map(int, input().split()))

# Create an instance of the HairGrowth class and calculate the days needed
hair_growth = HairGrowth()
result = hair_growth.daysToGrowHair(l, t, p)

# Print the result
print(result)