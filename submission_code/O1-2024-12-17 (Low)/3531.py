class Solution:
    def minDamage(self, power: int, damage: List[int], health: List[int]) -> int:
        # Number of enemies
        n = len(damage)
        
        # Calculate how many hits (T[i]) each enemy needs to die
        # T[i] = ceil(health[i] / power) = (health[i] + power - 1) // power
        hits_needed = [(damage[i], (health[i] + power - 1) // power) for i in range(n)]
        
        # Sort enemies in descending order by their damage
        hits_needed.sort(key=lambda x: x[0], reverse=True)
        
        # Precompute the suffix sums of damage. suffix_sum[k] = sum of damage from k-th to last
        suffix_sum = [0] * (n + 1)
        for i in range(n - 1, -1, -1):
            suffix_sum[i] = suffix_sum[i + 1] + hits_needed[i][0]
        
        total_damage = 0
        # For each enemy in sorted order, while we spend T(i) hits on the k-th enemy,
        # all enemies from k-th to end are still alive, so add T(i) * suffix_sum[k].
        for i in range(n):
            d, t = hits_needed[i]
            total_damage += t * suffix_sum[i]
        
        return total_damage