class Solution:
    def findMinimumTime(self, strength: List[int], K: int) -> int:
        n = len(strength)
        # Sort the locks by their strength to optimize the breaking process
        strength.sort()

        # Initialize the minimum time required
        min_time = 0

        # Initialize the current energy and factor
        current_energy = 0
        current_factor = 1

        for i in range(n):
            # Calculate the time needed to break the current lock
            while current_energy < strength[i]:
                current_energy += current_factor
                min_time += 1

            # Reset the energy and increase the factor after breaking the lock
            current_energy = 0
            current_factor += K

        return min_time