class Solution:
    def maxContainers(self, n: int, w: int, maxWeight: int) -> int:
        # Maximum containers limited by deck space
        by_space = n * n
        
        # Maximum containers limited by weight capacity
        by_weight = maxWeight // w
        
        # The ship can carry only as many containers as the stricter limit allows
        return min(by_space, by_weight)