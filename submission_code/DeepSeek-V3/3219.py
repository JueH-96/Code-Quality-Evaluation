class Solution:
    def lexicographicallySmallestArray(self, nums: List[int], limit: int) -> List[int]:
        n = len(nums)
        # Pair each number with its original index
        indexed_nums = [(nums[i], i) for i in range(n)]
        # Sort based on the number
        indexed_nums.sort()
        result = [0] * n
        # Group the numbers into clusters where the difference between consecutive elements is <= limit
        clusters = []
        current_cluster = [indexed_nums[0]]
        for i in range(1, n):
            if indexed_nums[i][0] - indexed_nums[i-1][0] <= limit:
                current_cluster.append(indexed_nums[i])
            else:
                clusters.append(current_cluster)
                current_cluster = [indexed_nums[i]]
        clusters.append(current_cluster)
        # For each cluster, sort the indices and assign the smallest numbers to the smallest indices
        for cluster in clusters:
            # Extract the indices and the numbers
            indices = [x[1] for x in cluster]
            numbers = [x[0] for x in cluster]
            # Sort the indices
            indices.sort()
            # Assign the sorted numbers to the sorted indices
            for i in range(len(indices)):
                result[indices[i]] = numbers[i]
        return result