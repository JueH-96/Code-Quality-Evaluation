class Solution:
    def minimumBoxes(self, apple: List[int], capacity: List[int]) -> int:
        total_apples = sum(apple)
        capacity.sort(reverse=True)
        current_sum = 0
        count = 0
        for cap in capacity:
            current_sum += cap
            count += 1
            if current_sum >= total_apples:
                return count
        return count