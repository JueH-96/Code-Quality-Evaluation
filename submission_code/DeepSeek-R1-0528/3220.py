class Solution:
    def countTestedDevices(self, batteryPercentages: List[int]) -> int:
        count = 0
        for battery in batteryPercentages:
            if battery - count > 0:
                count += 1
        return count