class Solution:
    def getMaxFunctionValue(self, receiver: List[int], k: int) -> int:
        n = len(receiver)
        max_val = 0
        for start_node in range(n):
            current_node = start_node
            current_sum = start_node
            for _ in range(k):
                current_node = receiver[current_node]
                current_sum += current_node
            max_val = max(max_val, current_sum)
        return max_val