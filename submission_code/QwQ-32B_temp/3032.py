class Solution:
    def getMaxFunctionValue(self, receiver: List[int], k: int) -> int:
        n = len(receiver)
        max_level = 40  # Sufficient since 2^40 > 1e10
        
        # Initialize up and sum_ tables
        up = [[0] * n for _ in range(max_level)]
        sum_ = [[0] * n for _ in range(max_level)]
        
        # Base case: level 0
        for x in range(n):
            up[0][x] = receiver[x]
            sum_[0][x] = receiver[x]
        
        # Fill tables for higher levels using binary lifting
        for j in range(1, max_level):
            for x in range(n):
                prev_node = up[j-1][x]
                up[j][x] = up[j-1][prev_node]
                sum_[j][x] = sum_[j-1][x] + sum_[j-1][prev_node]
        
        max_total = 0
        for x in range(n):
            current_node = x
            total = x
            remaining = k
            for j in reversed(range(max_level)):
                if (1 << j) <= remaining:
                    total += sum_[j][current_node]
                    current_node = up[j][current_node]
                    remaining -= (1 << j)
            if total > max_total:
                max_total = total
        
        return max_total