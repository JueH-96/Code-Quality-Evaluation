class Solution:
    def countTestedDevices(self, batteryPercentages: List[int]) -> int:
        tested = 0
        subtract = 0
        for battery in batteryPercentages:
            if battery - subtract > 0:
                tested += 1
                subtract += 1
        return tested