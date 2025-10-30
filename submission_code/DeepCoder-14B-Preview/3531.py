class Solution:
    def minDamage(self, power: int, damage: List[int], health: List[int]) -> int:
        enemies = []
        for d, h in zip(damage, health):
            k = (h + power - 1) // power  # Calculate the number of attacks needed
            enemies.append((d, k))
        
        # Sort enemies based on the ratio of damage per second to kills needed, in descending order
        enemies_sorted = sorted(enemies, key=lambda x: (x[0] / x[1]), reverse=True)
        
        pre = 0  # Cumulative time before attacking the current enemy
        total = 0
        
        for d, k in enemies_sorted:
            total += d * (pre + k)
            pre += k
        
        return total