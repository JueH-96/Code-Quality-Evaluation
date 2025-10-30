class Solution:
    def minOperations(self, s1: str, s2: str, x: int) -> int:
        n = len(s1)
        # Calculate the mismatch positions
        mismatch_positions = [i for i in range(n) if s1[i] != s2[i]]
        
        if len(mismatch_positions) % 2 != 0:
            return -1  # Odd number of mismatches cannot be resolved
        
        # We can use a greedy approach to minimize cost
        min_cost = 0
        i = 0
        while i < len(mismatch_positions):
            if i + 1 < len(mismatch_positions) and mismatch_positions[i + 1] == mismatch_positions[i] + 1:
                # Consecutive mismatch, use the second operation
                min_cost += 1
                i += 2
            else:
                # Non-consecutive mismatch, check if using the first operation is cheaper or possible
                if x <= 2:
                    # If the cost of the first operation is less than or equal to two consecutive second operations
                    if i + 1 < len(mismatch_positions):
                        min_cost += x
                        i += 2
                    else:
                        return -1  # Odd mismatch left at the end
                else:
                    # If the cost of two consecutive second operations is cheaper
                    if i + 3 < len(mismatch_positions) and mismatch_positions[i + 2] == mismatch_positions[i + 1] + 1:
                        # We have another pair of consecutive mismatches right after
                        min_cost += 2
                        i += 4
                    else:
                        # Use the first operation for this pair
                        if i + 1 < len(mismatch_positions):
                            min_cost += x
                            i += 2
                        else:
                            return -1  # Odd mismatch left at the end
        return min_cost