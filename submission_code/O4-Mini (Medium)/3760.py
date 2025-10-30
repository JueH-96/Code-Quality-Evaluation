from typing import List

class Solution:
    def assignElements(self, groups: List[int], elements: List[int]) -> List[int]:
        # Determine max group value to bound our sieve
        max_group_val = max(groups)
        n = len(elements)
        
        # Prepare the minimal index for each element value
        INF = 10**9
        # min_idx[x] = smallest j such that elements[j] == x
        min_idx = [INF] * (max_group_val + 1)
        for j, x in enumerate(elements):
            if x <= max_group_val:
                if j < min_idx[x]:
                    min_idx[x] = j
        
        # best_div[k] = smallest index j of an element that divides k
        best_div = [INF] * (max_group_val + 1)
        
        # Sieve over each distinct element value x to update its multiples
        for x in range(1, max_group_val + 1):
            j = min_idx[x]
            if j < INF:
                # x occurs in elements at index j; update multiples of x
                for multiple in range(x, max_group_val + 1, x):
                    if j < best_div[multiple]:
                        best_div[multiple] = j
        
        # Now assign for each group
        assigned = []
        for g in groups:
            idx = best_div[g]
            if idx < INF:
                assigned.append(idx)
            else:
                assigned.append(-1)
        
        return assigned

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    print(sol.assignElements([8,4,3,2,4], [4,2]))  # [0,0,-1,1,0]
    print(sol.assignElements([2,3,5,7], [5,3,3]))  # [-1,1,0,-1]
    print(sol.assignElements([10,21,30,41], [2,1]))# [0,1,0,1]