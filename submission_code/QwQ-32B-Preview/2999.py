class Solution:
    def canBeEqual(self, s1: str, s2: str) -> bool:
        # Generate all possible configurations for s1
        config1 = s1  # original
        config2 = s1[2] + s1[1] + s1[0] + s1[3]  # swap 0 and 2
        config3 = s1[0] + s1[3] + s1[2] + s1[1]  # swap 1 and 3
        config4 = s1[2] + s1[3] + s1[0] + s1[1]  # swap both 0<->2 and 1<->3
        
        # List of configurations
        configurations = [config1, config2, config3, config4]
        
        # Check if s2 matches any configuration of s1
        return s2 in configurations