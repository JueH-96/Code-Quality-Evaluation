from typing import List

class Solution:
    def assignElements(self, groups: List[int], elements: List[int]) -> List[int]:
        # Determine the maximum group size to bound our sieve
        max_group = max(groups)
        
        # Record the smallest index j for each element value v
        # Initialize with a large sentinel value
        INF = 10**9
        min_idx_of_val = [INF] * (max_group + 1)
        for j, v in enumerate(elements):
            if v <= max_group:
                # Keep only the smallest index for each value
                if j < min_idx_of_val[v]:
                    min_idx_of_val[v] = j
        
        # Build an array that for each possible group size g (1..max_group)
        # holds the smallest index j of an element that divides g
        min_idx_for_group = [INF] * (max_group + 1)
        
        # Sieve over multiples of each value v that actually appears
        for v in range(1, max_group + 1):
            j = min_idx_of_val[v]
            if j == INF:
                # no element equal to v appeared
                continue
            # mark all multiples of v
            for multiple in range(v, max_group + 1, v):
                if j < min_idx_for_group[multiple]:
                    min_idx_for_group[multiple] = j
        
        # Build the answer: for each group g, read off the precomputed index
        ans = []
        for g in groups:
            idx = min_idx_for_group[g]
            if idx == INF:
                ans.append(-1)
            else:
                ans.append(idx)
        
        return ans


# Example usage:
if __name__ == "__main__":
    sol = Solution()
    print(sol.assignElements([8,4,3,2,4], [4,2]))  # [0, 0, -1, 1, 0]
    print(sol.assignElements([2,3,5,7],   [5,3,3])) # [-1, 1, 0, -1]
    print(sol.assignElements([10,21,30,41],[2,1])) # [0, 1, 0, 1]