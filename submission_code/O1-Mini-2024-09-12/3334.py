class Solution:
    def minimumBoxes(self, apple: List[int], capacity: List[int]) -> int:
        total = sum(apple)
        capacity_sorted = sorted(capacity, reverse=True)
        selected_sum = 0
        count = 0
        for cap in capacity_sorted:
            selected_sum += cap
            count += 1
            if selected_sum >= total:
                break
        return count