class Solution:
    def countPairs(self, coordinates: List[List[int]], k: int) -> int:
        from collections import defaultdict

        # Dictionary to store the count of each coordinate
        coord_count = defaultdict(int)

        # Count the occurrences of each coordinate
        for x, y in coordinates:
            coord_count[(x, y)] += 1

        # Convert the dictionary to a list of tuples
        coord_list = list(coord_count.items())

        # Initialize the count of valid pairs
        count = 0

        # Iterate over all pairs of coordinates
        for i in range(len(coord_list)):
            for j in range(i + 1, len(coord_list)):
                (x1, y1), count1 = coord_list[i]
                (x2, y2), count2 = coord_list[j]

                # Calculate the distance between the points
                distance = (x1 ^ x2) + (y1 ^ y2)

                # If the distance is equal to k, add the product of their counts to the total count
                if distance == k:
                    count += count1 * count2

        return count