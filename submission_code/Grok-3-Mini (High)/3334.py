class Solution:
    def minimumBoxes(self, apple: List[int], capacity: List[int]) -> int:
        total_apples = sum(apple)
        sorted_cap = sorted(capacity, reverse=True)
        cum_sum = 0
        count = 0
        for cap in sorted_cap:
            cum_sum += cap
            count += 1
            if cum_sum >= total_apples:
                return count