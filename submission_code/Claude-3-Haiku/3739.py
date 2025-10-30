class Solution:
    def distanceSum(self, m: int, n: int, k: int) -> int:
        MOD = 10 ** 9 + 7
        
        # Helper function to calculate the Manhattan distance between two cells
        def manhattan_distance(x1, y1, x2, y2):
            return abs(x1 - x2) + abs(y1 - y2)
        
        # Initialize the total distance sum
        total_distance = 0
        
        # Iterate through all valid arrangements of pieces
        for arrangement in range(2 ** (m * n)):
            # Check if the current arrangement has exactly k pieces
            if bin(arrangement).count('1') == k:
                # Calculate the total distance for the current arrangement
                distance = 0
                for i in range(m * n):
                    if arrangement & (1 << i):
                        for j in range(i + 1, m * n):
                            if arrangement & (1 << j):
                                distance += manhattan_distance(i // n, i % n, j // n, j % n)
                total_distance += distance
        
        return total_distance % MOD