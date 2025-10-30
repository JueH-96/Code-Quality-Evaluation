class Solution:
    def distributeCandies(self, n: int, limit: int) -> int:
        def count_ways(total, num_children, max_per_child):
            if num_children == 1:
                return 1 if 0 <= total <= max_per_child else 0
            count = 0
            for i in range(min(total, max_per_child) + 1):
                count += count_ways(total - i, num_children - 1, max_per_child)
            return count

        return count_ways(n, 3, limit)