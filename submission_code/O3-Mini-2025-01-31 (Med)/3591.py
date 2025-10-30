from typing import List

class Solution:
    def shiftDistance(self, s: str, t: str, nextCost: List[int], previousCost: List[int]) -> int:
        # Precompute the cost for shifting forward and backward for each letter.
        # For a given starting letter (represented by an index 0 to 25),
        # forward[i][d] = cost to shift forward d times starting from letter with index i.
        # backward[i][d] = cost to shift backward d times starting from letter with index i.
        # We compute these for d from 0 to 26 (a full cycle).
        forward = [[0] * 27 for _ in range(26)]
        backward = [[0] * 27 for _ in range(26)]
        
        for i in range(26):
            # Forward shifting: cost for d shifts.
            for d in range(1, 27):
                fromLetter = (i + d - 1) % 26  # letter from which we shift
                forward[i][d] = forward[i][d - 1] + nextCost[fromLetter]
            
            # Backward shifting: cost for d shifts.
            for d in range(1, 27):
                fromLetter = (i - (d - 1)) % 26  # letter from which we shift backward
                backward[i][d] = backward[i][d - 1] + previousCost[fromLetter]
        
        total_cost = 0
        n = len(s)
        for i in range(n):
            src_idx = ord(s[i]) - ord('a')
            tgt_idx = ord(t[i]) - ord('a')
            
            # Calculate the number of shifts required in both directions.
            forward_shifts = (tgt_idx - src_idx) % 26
            backward_shifts = (src_idx - tgt_idx) % 26
            
            # Use our precomputed tables to get the cost.
            cost_forward = forward[src_idx][forward_shifts]
            cost_backward = backward[src_idx][backward_shifts]
            
            total_cost += min(cost_forward, cost_backward)
        
        return total_cost

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    
    # Example 1:
    s1 = "abab"
    t1 = "baba"
    nextCost1 = [100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    previousCost1 = [1, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    print(sol.shiftDistance(s1, t1, nextCost1, previousCost1))  # Expected output: 2

    # Example 2:
    s2 = "leet"
    t2 = "code"
    nextCost2 = [1] * 26
    previousCost2 = [1] * 26
    print(sol.shiftDistance(s2, t2, nextCost2, previousCost2))  # Expected output: 31