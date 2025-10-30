class Solution:
    def minimumBoxes(self, apple: List[int], capacity: List[int]) -> int:
        total_apples = sum(apple)
        capacity.sort()
        boxes = 0
        for c in reversed(capacity):
            if total_apples <= 0:
                break
            total_apples -= c
            boxes += 1
        return boxes