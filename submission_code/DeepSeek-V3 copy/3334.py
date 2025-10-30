class Solution:
    def minimumBoxes(self, apple: List[int], capacity: List[int]) -> int:
        total_apples = sum(apple)
        capacity.sort(reverse=True)
        count = 0
        current_capacity = 0
        for cap in capacity:
            if current_capacity >= total_apples:
                break
            current_capacity += cap
            count += 1
        return count