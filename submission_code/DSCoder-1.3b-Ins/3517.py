class Solution:
    def shortestDistanceAfterQueries(self, n: int, queries: List[List[int]]) -> List[int]:
        # Initialize the distance array
        distance = [1] * n

        # Initialize the count array
        count = [1] * n

        # Initialize the result array
        result = []

        # Loop through the queries
        for query in queries:
            i, j = query
            # If the query is a 'add' query
            if distance[i] != distance[j]:
                # Update the distance
                distance[i] = distance[j] = min(distance[i], distance[j])
                # Update the count
                count[i] = count[j] = 0
            else:
                # Update the count
                count[i] = count[j] += 1

        # Append the result to the result array
        result = [distance[i] + count[i] for i in range(n)]

        return result