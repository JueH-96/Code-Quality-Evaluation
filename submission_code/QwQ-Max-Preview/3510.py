class Solution:
    def maximumTotalSum(self, maximumHeight: List[int]) -> int:
        maximumHeight.sort()
        n = len(maximumHeight)
        
        # Check if any element is less than required minimum (i+1)
        for i in range(n):
            if maximumHeight[i] < i + 1:
                return -1
        
        total = 0
        prev = maximumHeight[-1]
        total += prev
        
        # Iterate from the second last element to the first
        for i in range(n - 2, -1, -1):
            current = min(maximumHeight[i], prev - 1)
            total += current
            prev = current
        
        return total