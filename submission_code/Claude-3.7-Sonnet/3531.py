class Solution:
    def minDamage(self, power: int, damage: List[int], health: List[int]) -> int:
        n = len(damage)
        
        # Calculate turns needed to kill each enemy
        turns_to_kill = [(health[i] + power - 1) // power for i in range(n)]
        
        # Sort enemies by damage/turns ratio (higher ratio first)
        enemies = sorted(range(n), key=lambda i: damage[i] / turns_to_kill[i], reverse=True)
        
        total_damage = 0
        cumulative_turns = 0
        
        # Calculate total damage
        for i in enemies:
            cumulative_turns += turns_to_kill[i]
            total_damage += damage[i] * cumulative_turns
        
        return total_damage