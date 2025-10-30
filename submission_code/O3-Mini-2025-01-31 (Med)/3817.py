class Solution:
    def maxContainers(self, n: int, w: int, maxWeight: int) -> int:
        # Maximum number of containers that the deck can hold (n x n cells)
        max_containers_on_deck = n * n
        
        # Maximum number of containers that can be loaded given the weight restriction
        max_containers_by_weight = maxWeight // w
        
        # The answer is the smaller of the two values
        return min(max_containers_on_deck, max_containers_by_weight)