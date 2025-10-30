class Solution:
    def maxSubarrays(self, n: int, conflictingPairs: List[List[int]]) -> int:
        # Initialize sorted_desc list for each position from 1 to n
        sorted_desc = [[] for _ in range(n + 1)]  # 0-based index up to n
        
        # Populate sorted_desc with conflicting pairs
        for a, b in conflictingPairs:
            u = min(a, b)
            v = max(a, b)
            if u < v:  # ensure u is strictly less than v
                sorted_desc[v].append(u)
        
        # Sort each list in descending order
        for x in range(1, n + 1):
            sorted_desc[x].sort(reverse=True)
        
        # Compute original_max_y and original_total
        original_max_y = [0] * (n + 1)
        original_total = 0
        
        for x in range(1, n + 1):
            if sorted_desc[x]:
                original_max_y[x] = sorted_desc[x][0]
            else:
                original_max_y[x] = 0
            
            if original_max_y[x] == 0:
                original_total += x
            else:
                original_total += (x - original_max_y[x])
        
        # Evaluate each candidate removal
        max_result = 1  # Minimum possible value
        
        for a, b in conflictingPairs:
            u = min(a, b)
            v = max(a, b)
            
            # Check if u is the current max in sorted_desc[v]
            if sorted_desc[v] and sorted_desc[v][0] == u:
                # Determine new_max after removal
                if len(sorted_desc[v]) == 1:
                    new_max = 0
                else:
                    new_max = sorted_desc[v][1]
                
                original_contribution = v - u
                if new_max == 0:
                    new_contribution = v
                else:
                    new_contribution = v - new_max
                
                delta = new_contribution - original_contribution
                candidate_total = original_total + delta
                
                if candidate_total > max_result:
                    max_result = candidate_total
            else:
                # Use original_total
                if original_total > max_result:
                    max_result = original_total
        
        return max_result