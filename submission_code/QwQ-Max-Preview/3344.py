class Solution:
    def minimumDistance(self, points: List[List[int]]) -> int:
        A = [x + y for x, y in points]
        B = [x - y for x, y in points]
        sorted_A = sorted(A)
        sorted_B = sorted(B)
        
        # Precompute for A
        maxA = sorted_A[-1]
        count_maxA = 0
        i = len(sorted_A) - 1
        while i >= 0 and sorted_A[i] == maxA:
            count_maxA += 1
            i -= 1
        second_maxA = maxA if count_maxA > 1 else sorted_A[-2]
        
        minA = sorted_A[0]
        count_minA = 0
        i = 0
        while i < len(sorted_A) and sorted_A[i] == minA:
            count_minA += 1
            i += 1
        second_minA = minA if count_minA > 1 else sorted_A[1]
        
        # Precompute for B
        maxB = sorted_B[-1]
        count_maxB = 0
        i = len(sorted_B) - 1
        while i >= 0 and sorted_B[i] == maxB:
            count_maxB += 1
            i -= 1
        second_maxB = maxB if count_maxB > 1 else sorted_B[-2]
        
        minB = sorted_B[0]
        count_minB = 0
        i = 0
        while i < len(sorted_B) and sorted_B[i] == minB:
            count_minB += 1
            i += 1
        second_minB = minB if count_minB > 1 else sorted_B[1]
        
        min_result = float('inf')
        for i in range(len(points)):
            a = A[i]
            b_val = B[i]
            
            # Compute new_maxA and new_minA
            if a == maxA:
                new_maxA_val = maxA if count_maxA > 1 else second_maxA
            else:
                new_maxA_val = maxA
            if a == minA:
                new_minA_val = minA if count_minA > 1 else second_minA
            else:
                new_minA_val = minA
            new_diffA = new_maxA_val - new_minA_val
            
            # Compute new_maxB and new_minB
            if b_val == maxB:
                new_maxB_val = maxB if count_maxB > 1 else second_maxB
            else:
                new_maxB_val = maxB
            if b_val == minB:
                new_minB_val = minB if count_minB > 1 else second_minB
            else:
                new_minB_val = minB
            new_diffB = new_maxB_val - new_minB_val
            
            current_max = max(new_diffA, new_diffB)
            if current_max < min_result:
                min_result = current_max
        
        return min_result