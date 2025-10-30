class Solution:
    def minTime(self, skill: List[int], mana: List[int]) -> int:
        n = len(skill)
        m = len(mana)
        
        # Track when each wizard will be free
        wizard_free = [0] * n
        
        for j in range(m):  # For each potion
            # For each wizard in sequence
            for i in range(n):
                if i == 0:
                    # First wizard starts when they're free
                    start_time = wizard_free[i]
                else:
                    # Other wizards start when both:
                    # 1. Previous wizard finished (wizard_free[i-1])
                    # 2. They are free (wizard_free[i])
                    start_time = max(wizard_free[i-1], wizard_free[i])
                
                # Calculate finish time
                work_time = skill[i] * mana[j]
                wizard_free[i] = start_time + work_time
        
        # Return when the last wizard finishes
        return wizard_free[n-1]