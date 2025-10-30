class Solution:
    def minimumBoxes(self, apple: List[int], capacity: List[int]) -> int:
        total_apples = sum(apple)
        capacity.sort(reverse=True)
        sum_cap = 0
        count = 0
        for cap in capacity:
            sum_cap += cap
            count += 1
            if sum_cap >= total_apples:
                return count
        return count  # This line is theoretically unreachable due to problem constraints