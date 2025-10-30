class Solution:
    def maximumTotalDamage(self, power: List[int]) -> int:
        from collections import defaultdict
        
        # Create a dictionary to count occurrences of each power value
        power_count = defaultdict(int)
        for p in power:
            power_count[p] += 1
        
        # Sort the unique power values
        unique_powers = sorted(power_count.keys())
        
        # Initialize a dictionary to store the maximum damage up to each power value
        max_damage = defaultdict(int)
        
        for p in unique_powers:
            # Calculate the maximum damage if we include this power
            include_damage = p * power_count[p]
            if p - 3 in max_damage:
                include_damage += max_damage[p - 3]
            elif p - 2 in max_damage:
                include_damage += max_damage[p - 2]
            
            # Calculate the maximum damage if we exclude this power
            exclude_damage = max_damage[p - 1] if p - 1 in max_damage else 0
            
            # Store the maximum damage up to this power
            max_damage[p] = max(include_damage, exclude_damage)
        
        # The result is the maximum damage we can achieve
        return max(max_damage.values())