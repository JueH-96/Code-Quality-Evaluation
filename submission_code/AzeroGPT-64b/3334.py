class Solution:
    def minimumBoxes(self, apple: List[int], capacity: List[int]) -> int:
        total_apples = sum(apple)
        capacity.sort(reverse=True)
        
        count = 0
        for cap in capacity:
            if total_apples > 0:
                total_apples -= min(total_apples, cap)
                count += 1
        
        return count