from typing import List

class Solution:
    def maximumTotalSum(self, maximumHeight: List[int]) -> int:
        n = len(maximumHeight)
        
        # 1. Sort the maximum heights.
        # Let the sorted maximum heights be m_0 <= m_1 <= ... <= m_{n-1}.
        # This takes O(N log N) time.
        m = sorted(maximumHeight)
        
        # 2. Check the necessary and sufficient condition for feasibility:
        # We need to assign n distinct positive integer heights h_0, ..., h_{n-1}
        # such that h_i <= original_maximumHeight[i] for all i.
        # Let the sorted assigned heights be v_0 < v_1 < ... < v_{n-1}.
        # A valid assignment exists if and only if v_i <= m_i for all i,
        # where m_i are the sorted maximum heights.
        # To have n distinct positive integers, the smallest possible values are 1, 2, ..., n.
        # The i-th smallest distinct positive integer is at least i + 1.
        # Thus, for a valid assignment to exist, we must be able to assign a height >= i + 1
        # to the tower with the i-th smallest maximum height m[i]. This requires m[i] >= i + 1 for all i.
        # If this condition is not met, it's impossible to find such an assignment.
        # This check takes O(N) time.
        for i in range(n):
            # Check if the i-th smallest max height can accommodate the i-th smallest distinct positive height (i+1)
            if m[i] < i + 1:
                return -1
                
        # 3. Calculate the maximum possible total sum.
        # We need to find 1 <= v_0 < v_1 < ... < v_{n-1} such that v_i <= m_i for all i,
        # maximizing sum(v_i).
        # We can find the optimal v_i values by working backward from the largest:
        # v_{n-1} = m[n-1] (to maximize the largest value within its constraint)
        # v_{i} = min(m[i], v_{i+1} - 1) for i from n-2 down to 0.
        # This ensures v_i <= m[i] and v_i < v_{i+1}.
        # The feasibility check m[i] >= i+1 ensures that v_i calculated this way is always >= i+1,
        # hence positive and maintaining the strict inequality v_i >= v_{i-1} + 1.
        
        total_sum = 0
        
        # Calculate v_{n-1} and add to sum
        # This height corresponds to v_{n-1} in the sorted assigned heights {v_0, ..., v_{n-1}}
        current_h = m[n-1] 
        total_sum += current_h
        
        # Calculate v_i for i = n-2 down to 0 and add to sum.
        # 'current_h' in each iteration i holds the value of v_{i+1} from the previous step.
        # This loop takes O(N) time.
        for i in range(n - 2, -1, -1):
            # Calculate v_i = min(m[i], v_{i+1} - 1)
            # m[i] is the maximum allowed height for this tower (in sorted order)
            # current_h - 1 is the maximum allowed height to be strictly less than the next height (v_{i+1})
            current_h = min(m[i], current_h - 1)
            total_sum += current_h
        
        return total_sum