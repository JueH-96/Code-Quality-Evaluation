class Solution:
    def minimumBoxes(self, apple: List[int], capacity: List[int]) -> int:
        total_apples = sum(apple)
        capacity.sort(reverse=True)
        
        boxes = 0
        curr_capacity = 0
        
        for cap in capacity:
            if curr_capacity >= total_apples:
                break
            curr_capacity += cap
            boxes += 1
            
        return boxes