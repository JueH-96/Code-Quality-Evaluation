class Solution:
    def minimumBoxes(self, apple: List[int], capacity: List[int]) -> int:
        total_apples = sum(apple)
        capacity.sort(reverse=True)
        boxes_used = 0
        
        for cap in capacity:
            if total_apples > 0:
                total_apples -= cap
                boxes_used += 1
            else:
                break
                
        return boxes_used