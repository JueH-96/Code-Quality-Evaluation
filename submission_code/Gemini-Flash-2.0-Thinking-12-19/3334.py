class Solution:
    def minimumBoxes(self, apple: List[int], capacity: List[int]) -> int:
        total_apples = sum(apple)
        capacity.sort(reverse=True)
        boxes_count = 0
        current_capacity = 0
        for cap in capacity:
            current_capacity += cap
            boxes_count += 1
            if current_capacity >= total_apples:
                return boxes_count
        return boxes_count