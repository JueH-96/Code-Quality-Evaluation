class Solution:
    def maximumTotalCost(self, nums: List[int]) -> int:
        n = len(nums)
        
        def calculate_cost(arr):
            cost = 0
            for i in range(len(arr)):
                cost += arr[i] * (1 if i % 2 == 0 else -1)
            return cost

        max_cost = calculate_cost(nums)
        
        for i in range(1, 1 << (n - 1)):
            partitions = []
            current_partition = []
            for j in range(n):
                current_partition.append(nums[j])
                if (i >> j) & 1 and j < n -1:
                    partitions.append(current_partition)
                    current_partition = []
            partitions.append(current_partition)
            
            total_cost = 0
            for partition in partitions:
                total_cost += calculate_cost(partition)
            max_cost = max(max_cost, total_cost)

        return max_cost