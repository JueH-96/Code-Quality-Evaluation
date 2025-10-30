import bisect

class Solution:
    def shortestMatchingSubstring(self, s: str, p: str) -> int:
        # Split the pattern into left, middle, right parts
        first_star = p.find('*')
        second_star = p.find('*', first_star + 1)
        left_part = p[:first_star]
        middle_part = p[first_star+1:second_star]
        right_part = p[second_star+1:]
        
        left_empty = (left_part == "")
        middle_empty = (middle_part == "")
        right_empty = (right_part == "")
        
        # Helper function to find all starting indices of a substring in s
        def find_all_occurrences(sub):
            indices = []
            if not sub:
                return indices  # empty handled separately
            start = 0
            while True:
                pos = s.find(sub, start)
                if pos == -1:
                    break
                indices.append(pos)
                start = pos + 1
            return indices
        
        # Prepare the lists for each part
        L_list = []
        if left_empty:
            L_list = [0]
        else:
            L_list = find_all_occurrences(left_part)
        
        M_list = []
        if not middle_empty:
            M_list = find_all_occurrences(middle_part)
        
        R_list = []
        if not right_empty:
            R_list = find_all_occurrences(right_part)
        
        min_length = float('inf')
        
        # Iterate over each possible L_start in L_list
        for L in L_list:
            L_end = L + len(left_part)
            
            # Determine M_start and M_end_pos
            if middle_empty:
                M_start = L_end
                M_end_pos = M_start  # since middle is empty
            else:
                # Find first M in M_list >= L_end
                idx = bisect.bisect_left(M_list, L_end)
                if idx == len(M_list):
                    continue  # no valid M found
                M_start = M_list[idx]
                M_end_pos = M_start + len(middle_part) - 1
            
            # Determine R_start
            if right_empty:
                # R can start at M_end_pos (since right is empty)
                R_start = M_end_pos
            else:
                # Find first R in R_list >= (M_end_pos + 1)
                target = M_end_pos + 1
                idx = bisect.bisect_left(R_list, target)
                if idx == len(R_list):
                    continue
                R_start = R_list[idx]
            
            # Calculate the length
            if right_empty:
                length = R_start - L
            else:
                length = (R_start + len(right_part)) - L
            
            if length < min_length:
                min_length = length
        
        if min_length == float('inf'):
            return -1
        else:
            return min_length