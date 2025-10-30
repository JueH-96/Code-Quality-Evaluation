class Solution:
    def minDamage(self, power: int, damage: List[int], health: List[int]) -> int:
        n = len(damage)
        
        # Calculate time to kill each enemy
        time_to_kill = []
        for i in range(n):
            # Using integer arithmetic for ceiling division
            time_to_kill.append((health[i] + power - 1) // power)
        
        # Create list of indices
        indices = list(range(n))
        
        # Sort by damage/time_to_kill ratio in descending order
        # Kill enemies with highest damage per time unit first
        indices.sort(key=lambda i: damage[i] / time_to_kill[i], reverse=True)
        
        # Calculate total damage
        total_damage = 0
        alive_damage = sum(damage)  # Total damage per second from all alive enemies
        
        for i in indices:
            # This enemy takes time_to_kill[i] seconds to kill
            # During this time, Bob takes alive_damage per second
            total_damage += alive_damage * time_to_kill[i]
            # After killing this enemy, reduce the alive_damage
            alive_damage -= damage[i]
        
        return total_damage