class Solution:
    def minDamage(self, power: int, damage: List[int], health: List[int]) -> int:
        # Calculate turns needed to kill each enemy
        turns = [(h + power - 1) // power for h in health]
        
        # Calculate total damage from each enemy
        total_damage = 0
        for i in range(len(damage)):
            total_damage += damage[i] * turns[i]
            
        # For each enemy, calculate damage saved if killed earlier
        savings = []
        for i in range(len(damage)):
            savings.append((damage[i], turns[i], i))
            
        # Sort by damage per turn (descending)
        savings.sort(key=lambda x: (-x[0], x[1]))
        
        # Track turns used
        used_turns = 0
        
        # For each enemy in order of damage per turn
        for dmg, turn, idx in savings:
            # All enemies before this one take 'used_turns' extra turns to kill
            total_damage -= dmg * used_turns
            # Add turns needed for this enemy
            used_turns += turn
            
        return total_damage