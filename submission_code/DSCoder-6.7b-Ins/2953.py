class Solution:
    def countPairs(self, coordinates: List[List[int]], k: int) -> int:
        # Initialize a counter
        counter = 0
        # Iterate over the coordinates
        for i in range(len(coordinates)):
            for j in range(i+1, len(coordinates)):
                # Calculate the distance between the two points
                distance = (coordinates[i][0] ^ coordinates[j][0]) + (coordinates[i][1] ^ coordinates[j][1])
                # If the distance is equal to k, increment the counter
                if distance == k:
                    counter += 1
        # Return the counter
        return counter