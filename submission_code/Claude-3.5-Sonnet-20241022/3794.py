class Solution:
    def minTime(self, skill: List[int], mana: List[int]) -> int:
        def can_brew_all_potions(target_time):
            # For each potion, track the earliest possible start time
            earliest_start = 0
            
            # Try to brew each potion
            for j in range(len(mana)):
                current_time = earliest_start
                
                # Calculate when each wizard finishes with current potion
                for i in range(len(skill)):
                    # Time taken by current wizard for this potion
                    wizard_time = skill[i] * mana[j]
                    
                    # When will this wizard finish?
                    finish_time = current_time + wizard_time
                    
                    # If any wizard takes longer than target_time, brewing is impossible
                    if finish_time > target_time:
                        return False
                    
                    # Next wizard must start exactly when current wizard finishes
                    current_time = finish_time
                
                # The next potion can't start before the earliest time
                # that allows all wizards to be free
                earliest_start = max(earliest_start + skill[0] * mana[j], 
                                   current_time - sum(skill[i] * mana[j] for i in range(len(skill))))
            
            return True
        
        # Binary search for the minimum possible time
        left = 0
        # Upper bound: maximum time if potions are brewed sequentially
        right = sum(max(skill[i] * mana[j] for i in range(len(skill))) 
                   for j in range(len(mana))) * len(skill)
        
        while left < right:
            mid = (left + right) // 2
            if can_brew_all_potions(mid):
                right = mid
            else:
                left = mid + 1
                
        return left