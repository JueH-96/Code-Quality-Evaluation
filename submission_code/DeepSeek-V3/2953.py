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
            # Iterate through possible x2 and y2 such that (x1 XOR x2) + (y1 XOR y2) = k
            # Since x1 XOR x2 = a and y1 XOR y2 = b, and a + b = k
            # We can iterate over all possible a and b such that a + b = k
            for a in range(k + 1):
                b = k - a
                # Calculate x2 and y2
                x2 = x1 ^ a
                y2 = y1 ^ b
                # Check if (x2, y2) exists in the frequency map
                if (x2, y2) in freq:
                    # If x1 == x2 and y1 == y2, then it's the same point
                    if x1 == x2 and y1 == y2:
                        # The number of pairs is cnt1 * (cnt1 - 1) // 2
                        count += cnt1 * (cnt1 - 1) // 2
                    else:
                        # Otherwise, add cnt1 * cnt2 to the count
                        # To avoid double counting, ensure that (x1, y1) < (x2, y2)
                        if (x1, y1) < (x2, y2):
                            count += cnt1 * freq[(x2, y2)]
        
        return count