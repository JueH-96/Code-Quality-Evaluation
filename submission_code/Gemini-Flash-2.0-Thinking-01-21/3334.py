class Solution:
    def minimumBoxes(self, apple: List[int], capacity: List[int]) -> int:
        total_apples = sum(apple)
        capacity.sort(reverse=True)
        current_capacity = 0
        boxes_count = 0
        for box_capacity in capacity:
            current_capacity += box_capacity
            boxes_count += 1
            if current_capacity >= total_apples:
                return boxes_count
        return boxes_count