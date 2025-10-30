from typing import List

class Solution:
    def countTestedDevices(self, batteryPercentages: List[int]) -> int:
        n = len(batteryPercentages)
        tested_count = 0
        
        # Process each device in order
        for i in range(n):
            if batteryPercentages[i] > 0:
                tested_count += 1
                # Decrease battery for all devices from i+1 to end
                for j in range(i + 1, n):
                    batteryPercentages[j] = max(0, batteryPercentages[j] - 1)
        
        return tested_count