import heapq

class Solution:
    def minDamage(self, power: int, damage: List[int], health: List[int]) -> int:
        # Create a list of tuples with negative (damage + power) for heap sorting
        combined = []
        for d, h in zip(damage, health):
            combined.append((- (d + power), d, h))
        
        # Use a min-heap to sort in ascending order, which gives us the largest (d+power) when popped
        heapq.heapify(combined)
        
        total_damage = 0
        
        while combined:
            # Pop the smallest element, which is the largest (d+power) when considering the negative
            d_plus_p, d, h = heapq.heappop(combined)
            d_plus_p = -d_plus_p  # Revert the negative
            
            # Calculate the number of attacks needed
            t = (h + d_plus_p - 1) // d_plus_p
            
            # Add to the total damage
            total_damage += d_plus_p * t
        
        return total_damage