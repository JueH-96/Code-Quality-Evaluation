class Solution:
    def getMaxFunctionValue(self, receiver: List[int], k: int) -> int:
        n = len(receiver)
        max_value = 0
        for start_node in range(n):
            current_node = start_node
            current_value = start_node
            for _ in range(k):
                current_node = receiver[current_node]
                current_value += current_node
            max_value = max(max_value, current_value)
        return max_value