class Solution:
    def countTestedDevices(self, batteryPercentages: List[int]) -> int:
        count = 0
        dec = 0
        for perc in batteryPercentages:
            if perc - dec > 0:
                count += 1
                dec += 1
        return count