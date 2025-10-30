class Solution:
    def minimumBoxes(self, apple: List[int], capacity: List[int]) -> int:
        total_apples = sum(apple)
        capacity.sort(reverse=True)
        boxes_used = 0
        for cap in capacity:
            total_apples -= cap
            boxes_used += 1
            if total_apples <= 0:
                return boxes_used