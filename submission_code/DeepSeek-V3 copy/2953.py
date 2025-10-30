from collections import defaultdict

class Solution:
    def countPairs(self, coordinates: List[List[int]], k: int) -> int:
        # Create a frequency map to count occurrences of each (x, y) pair
        freq = defaultdict(int)
        for x, y in coordinates:
            freq[(x, y)] += 1
        
        # Initialize the count of valid pairs
        count = 0
        
        # Iterate through each unique (x1, y1) pair
        for (x1, y1), cnt1 in freq.items():
            # Iterate through each possible a and b such that a + b = k
            for a in range(k + 1):
                b = k - a
                # Calculate x2 and y2 based on the XOR properties
                x2 = x1 ^ a
                y2 = y1 ^ b
                # Check if (x2, y2) exists in the frequency map
                if (x2, y2) in freq:
                    # If (x1, y1) == (x2, y2), we need to handle the case where the same pair is chosen
                    if x1 == x2 and y1 == y2:
                        # The number of ways to choose 2 from cnt1 is cnt1 * (cnt1 - 1) // 2
                        count += cnt1 * (cnt1 - 1) // 2
                    else:
                        # Otherwise, multiply the counts of the two pairs
                        # To avoid double counting, ensure that we only count pairs where (x1, y1) < (x2, y2)
                        if (x1, y1) < (x2, y2):
                            count += cnt1 * freq[(x2, y2)]
        
        return count