class Solution:
    def maxTotalReward(self, rewardValues: List[int]) -> int:
        rewardValues.sort()
        possible = {0}
        max_sum = 0
        for x in rewardValues:
            temp = set()
            for s in possible:
                if s < x:
                    new_s = s + x
                    temp.add(new_s)
            possible.update(temp)
            if temp:
                current_max = max(temp)
                if current_max > max_sum:
                    max_sum = current_max
        return max_sum