class Solution:
    def minimumBoxes(self, apple: List[int], capacity: List[int]) -> int:
        total_apples = sum(apple)
        capacity.sort(reverse=True)
        boxes_used = 0
        
        for box in capacity:
            total_apples -= box
            boxes_used += 1
            if total_apples <= 0:
                return boxes_used