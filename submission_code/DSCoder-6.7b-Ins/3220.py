class Solution:
    def countTestedDevices(self, batteryPercentages: List[int]) -> int:
        n = len(batteryPercentages)
        tested_devices = 0
        for i in range(n):
            if batteryPercentages[i] > 0:
                tested_devices += 1
                batteryPercentages[i] -= 1
                for j in range(i+1, n):
                    batteryPercentages[j] = max(0, batteryPercentages[j]-1)
        return tested_devices