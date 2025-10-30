class Solution:
    def minimumBoxes(self, apple: List[int], capacity: List[int]) -> int:
        total_apples = sum(apple)
        capacity.sort(reverse=True)
        
        boxes_used = 0
        current_capacity = 0
        
        for i in range(len(capacity)):
            current_capacity += capacity[i]
            boxes_used += 1
            if current_capacity >= total_apples:
                return boxes_used
        
        return boxes_used