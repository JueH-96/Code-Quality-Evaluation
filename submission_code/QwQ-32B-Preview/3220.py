from typing import List

class Solution:
    def countTestedDevices(self, batteryPercentages: List[int]) -> int:
        tested_so_far = 0
        for battery in batteryPercentages:
            effective_battery = battery - tested_so_far
            if effective_battery > 0:
                tested_so_far += 1
        return tested_so_far