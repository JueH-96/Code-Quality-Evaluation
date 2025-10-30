from typing import List

class Solution:
    def findMinimumTime(self, strength: List[int], K: int) -> int:
        n = len(strength)
        # Sort the strength array in descending order to break the strongest locks first
        strength.sort(reverse=True)

        total_time = 0
        current_factor = 1

        for s in strength:
            current_energy = 0
            minutes = 0
            # Increase energy until it is enough to break the current lock
            while current_energy < s:
                current_energy += current_factor
                minutes += 1
            # Add the time taken to break this lock
            total_time += minutes
            # Increase the factor by K for the next lock
            current_factor += K

        return total_time