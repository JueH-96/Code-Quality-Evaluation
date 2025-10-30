class Solution:
    def maximumPoints(self, enemyEnergies: List[int], currentEnergy: int) -> int:
        # Sort enemies by energy
        enemies = sorted(enemyEnergies)
        
        # The enemy with the lowest energy (most efficient for Operation 1)
        min_energy = enemies[0]
        
        # Maximum points we can get
        max_points = 0
        
        # Try different initial strategies
        for initial_attacks in range(currentEnergy // min_energy + 1):
            # Simulate the process
            energy = currentEnergy - (initial_attacks * min_energy)
            points = initial_attacks
            
            # Create a list of remaining enemies (for potential Operation 2)
            remaining = enemies.copy()
            marked = set()
            
            # Continue until we can't make progress
            while True:
                # Try to use Operation 2 on the highest energy enemy
                if points > 0:
                    # Find highest energy unmarked enemy
                    best_idx = -1
                    best_energy = 0
                    for i, e in enumerate(remaining):
                        if i not in marked and e > best_energy:
                            best_energy = e
                            best_idx = i
                    
                    if best_idx != -1 and best_energy > 0:
                        energy += best_energy
                        points -= 1
                        marked.add(best_idx)
                        continue
                
                # Use all remaining energy for Operation 1 on the lowest energy enemy
                additional_points = energy // min_energy
                if additional_points > 0:
                    points += additional_points
                    energy %= min_energy
                    continue
                
                # If we can't do either operation, we're done
                break
            
            max_points = max(max_points, points)
        
        return max_points