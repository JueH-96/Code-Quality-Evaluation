class Solution:
    def minimumDistance(self, points: List[List[int]]) -> int:
        """
        We want to remove exactly one point so that the maximum Manhattan distance
        among the remaining points is as small as possible.

        Recall that for any two points (x1, y1), (x2, y2), the Manhattan distance is:
            dist = |x1 - x2| + |y1 - y2|
        This can be rewritten using transforms:
            A = x + y
            B = x - y
        Then the Manhattan distance between two points is the maximum of:
            |A1 - A2| or |B1 - B2|  (where A1 = x1+y1, etc.)
        Hence, the maximum Manhattan distance among a set of points is
        max( (max(A) - min(A)), (max(B) - min(B)) ).

        We will:
         1) Compute arrays A[i] = x_i + y_i, B[i] = x_i - y_i.
         2) Sort indices of points by A and by B. 
         3) Let M_A = max(A), m_A = min(A), M_B = max(B), m_B = min(B).
            The overall max distance is max(M_A - m_A, M_B - m_B).
         4) When we remove one point p, the new maximum distance is
            max( M_A_except_p - m_A_except_p, M_B_except_p - m_B_except_p ).
            Where M_A_except_p is the max A value among all points except p, etc.
         5) Because only one point is removed, if that point is not the unique
            largest (or smallest), the relevant max/min won't change. Otherwise,
            it may shift to the second largest (or second smallest).
            We handle duplicates carefully.
         6) We compute the new max distance for each possible removed point,
            then take the minimum over all. This is done efficiently by
            looking up in O(1) the "next max/min" from sorted arrays (and
            checking if the largest/smallest is duplicated).
        """

        import sys
        input_data = points
        
        n = len(input_data)
        # Precompute A, B for each point
        A = [x + y for x, y in input_data]
        B = [x - y for x, y in input_data]
        
        # Sort indices by A and by B
        # We only need to keep track of (value, index) pairs
        sortedA = sorted([(val, i) for i, val in enumerate(A)])
        sortedB = sorted([(val, i) for i, val in enumerate(B)])
        
        # Extract extremes for A
        A_min_val = sortedA[0][0]        # smallest A
        A_min_idx = sortedA[0][1]
        A_second_min_val = sortedA[1][0] # second smallest A
        A_max_val = sortedA[-1][0]       # largest A
        A_max_idx = sortedA[-1][1]
        A_second_max_val = sortedA[-2][0]# second largest A
        
        # Count how many times the largest A appears (for tie-check)
        # and how many times the smallest A appears.
        # We only need to check the last two and first two elements in sortedA 
        # for duplicates since the array is sorted.
        largestA_count = 1
        if n >= 2 and sortedA[-1][0] == sortedA[-2][0]:
            # There's at least a tie for the largest
            largestA_count += 1
        
        smallestA_count = 1
        if n >= 2 and sortedA[0][0] == sortedA[1][0]:
            # There's a tie for the smallest
            smallestA_count += 1
        
        # Extract extremes for B
        B_min_val = sortedB[0][0]
        B_min_idx = sortedB[0][1]
        B_second_min_val = sortedB[1][0]
        B_max_val = sortedB[-1][0]
        B_max_idx = sortedB[-1][1]
        B_second_max_val = sortedB[-2][0]
        
        largestB_count = 1
        if n >= 2 and sortedB[-1][0] == sortedB[-2][0]:
            largestB_count += 1
        
        smallestB_count = 1
        if n >= 2 and sortedB[0][0] == sortedB[1][0]:
            smallestB_count += 1
        
        # For each point i, determine:
        #   newAmax = max A value except i
        #   newAmin = min A value except i
        #   newBmax = max B value except i
        #   newBmin = min B value except i
        # Then compute newDist = max(newAmax - newAmin, newBmax - newBmin)
        # We'll track the global minimum of these newDist values.
        
        INF = 10**18
        ans = INF
        
        for i in range(n):
            # 1) newAmax
            # if A[i] != A_max_val, the largest doesn't change => newAmax = A_max_val
            # else we check if there's a tie or not
            if A[i] != A_max_val:
                newAmax = A_max_val
            else:
                # if A[i] == A_max_val
                if largestA_count > 1:
                    # there's a tie for the largest => removing one won't change it
                    newAmax = A_max_val
                else:
                    # there's only one such largest => now secondLargest
                    newAmax = A_second_max_val
            
            # 2) newAmin
            if A[i] != A_min_val:
                newAmin = A_min_val
            else:
                # A[i] == A_min_val
                if smallestA_count > 1:
                    # tie => remains
                    newAmin = A_min_val
                else:
                    newAmin = A_second_min_val
            
            # 3) newBmax
            if B[i] != B_max_val:
                newBmax = B_max_val
            else:
                if largestB_count > 1:
                    newBmax = B_max_val
                else:
                    newBmax = B_second_max_val
            
            # 4) newBmin
            if B[i] != B_min_val:
                newBmin = B_min_val
            else:
                if smallestB_count > 1:
                    newBmin = B_min_val
                else:
                    newBmin = B_second_min_val
            
            # new distance
            candidate_dist = max(newAmax - newAmin, newBmax - newBmin)
            ans = min(ans, candidate_dist)
        
        return ans