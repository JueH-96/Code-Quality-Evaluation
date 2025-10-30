class Solution:
    def countTestedDevices(self, batteryPercentages: List[int]) -> int:
        count = 0
        for bat in batteryPercentages:
            if bat - count > 0:
                count += 1
        return count