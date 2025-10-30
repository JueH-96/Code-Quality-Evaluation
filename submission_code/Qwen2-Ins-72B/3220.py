class Solution:
    def countTestedDevices(self, batteryPercentages: List[int]) -> int:
        tested = 0
        for percentage in batteryPercentages:
            if percentage > tested:
                tested += 1
        return tested