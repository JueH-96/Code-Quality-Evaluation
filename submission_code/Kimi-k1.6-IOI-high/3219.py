class Solution:
    def lexicographicallySmallestArray(self, nums: List[int], limit: int) -> List[int]:
        n = len(nums)
        sorted_pairs = sorted(zip(nums, range(n)), key=lambda x: x[0])
        res = [0] * n
        
        if n == 0:
            return res
        
        clusters = []
        current_cluster = [sorted_pairs[0]]
        for i in range(1, n):
            if sorted_pairs[i][0] - sorted_pairs[i-1][0] <= limit:
                current_cluster.append(sorted_pairs[i])
            else:
                clusters.append(current_cluster)
                current_cluster = [sorted_pairs[i]]
        clusters.append(current_cluster)
        
        for cluster in clusters:
            values = [x[0] for x in cluster]
            indices = [x[1] for x in cluster]
            values_sorted = sorted(values)
            indices_sorted = sorted(indices)
            for i in range(len(indices_sorted)):
                res[indices_sorted[i]] = values_sorted[i]
        
        return res