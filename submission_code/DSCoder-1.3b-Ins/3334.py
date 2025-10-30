class Solution:
    def minimumBoxes(self, apple: List[int], capacity: List[int]) -> int:
        n = len(apple)
        prefix_sum = [0] * (n + 1)
        for i in range(n):
            prefix_sum[i + 1] = prefix_sum[i] + apple[i]

        max_capacity = capacity[0]
        for i in range(1, m):
            max_capacity = max(max_capacity, capacity[i])

        res = prefix_sum[n]
        for i in range(n):
            if prefix_sum[i] >= max_capacity:
                res += prefix_sum[i] - max_capacity
                max_capacity = capacity[i]

        return res