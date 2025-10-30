class Solution:
    def countTestedDevices(self, batteryPercentages: List[int]) -> int:
        tested_count = 0
        
        for percentage in batteryPercentages:
            # Current effective battery = original battery - number of previously tested devices
            if percentage - tested_count > 0:
                tested_count += 1
        
        return tested_count