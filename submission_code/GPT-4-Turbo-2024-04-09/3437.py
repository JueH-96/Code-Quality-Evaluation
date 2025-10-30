class Solution:
    def maximumTotalDamage(self, power: List[int]) -> int:
        from collections import defaultdict
        
        # Dictionary to store the sum of damages for each unique power value
        damage_dict = defaultdict(int)
        
        # Fill the dictionary with the sum of damages for each power value
        for p in power:
            damage_dict[p] += p
        
        # Sort the keys of the dictionary to process them in increasing order
        sorted_powers = sorted(damage_dict.keys())
        
        # Dictionary to store the maximum damage up to each power value
        dp = {}
        
        # Initialize the first power's maximum damage
        dp[sorted_powers[0]] = damage_dict[sorted_powers[0]]
        
        # Fill dp for each power value
        for i in range(1, len(sorted_powers)):
            current_power = sorted_powers[i]
            current_damage = damage_dict[current_power]
            
            # Option 1: Take the damage of the current power
            # Find the last power that does not conflict with the current power
            last_non_conflicting_index = -1
            for j in range(i - 1, -1, -1):
                if sorted_powers[j] < current_power - 2:
                    last_non_conflicting_index = j
                    break
            
            if last_non_conflicting_index == -1:
                # If no non-conflicting power, just take the current damage
                option1 = current_damage
            else:
                # Otherwise, add the current damage to the best result up to the last non-conflicting power
                option1 = dp[sorted_powers[last_non_conflicting_index]] + current_damage
            
            # Option 2: Do not take the current power, continue with the best result from the previous power
            option2 = dp[sorted_powers[i - 1]]
            
            # The best result for the current power is the maximum of these two options
            dp[current_power] = max(option1, option2)
        
        # The answer is the maximum damage calculated up to the last power
        return dp[sorted_powers[-1]]