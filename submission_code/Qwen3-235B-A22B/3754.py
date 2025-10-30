import heapq

class Solution:
    def maxDistance(self, s: str, k: int) -> int:
        max_total = 0
        
        for expr_idx in range(4):
            prefix_sum = 0
            gain_sum = 0
            heap = []
            current_max = 0
            
            for c in s:
                # Determine contribution based on the current expression
                if expr_idx == 0:
                    ci = 1 if c in ('N', 'E') else -1
                elif expr_idx == 1:
                    ci = 1 if c in ('S', 'E') else -1
                elif expr_idx == 2:
                    ci = 1 if c in ('N', 'W') else -1
                else:  # expr_idx == 3
                    ci = 1 if c in ('S', 'W') else -1
                
                # Update prefix sum
                prefix_sum += ci
                gain_i = 1 - ci
                
                # Handle gain_i if positive
                if gain_i > 0:
                    if len(heap) < k:
                        heapq.heappush(heap, gain_i)
                        gain_sum += gain_i
                    else:
                        if gain_i > heap[0]:
                            popped = heapq.heappushpop(heap, gain_i)
                            gain_sum += (gain_i - popped)
                
                # Update current maximum for the expression
                current_total = prefix_sum + gain_sum
                if current_total > current_max:
                    current_max = current_total
            
            # Update the global maximum
            if current_max > max_total:
                max_total = current_max
        
        return max_total