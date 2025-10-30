class Solution:
    def minimumBoxes(self, apple: List[int], capacity: List[int]) -> int:
        total_apples = sum(apple)
        capacity.sort(reverse=True)
        
        boxes_needed = 0
        apples_distributed = 0
        
        for box_capacity in capacity:
            if apples_distributed >= total_apples:
                break
            apples_distributed += box_capacity
            boxes_needed += 1
        
        return boxes_needed