class Solution:
    def minOperations(self, s1: str, s2: str, x: int) -> int:
        n = len(s1)
        diff_indices = []
        for i in range(n):
            if s1[i] != s2[i]:
                diff_indices.append(i)
        if len(diff_indices) % 2 != 0:
            return -1
        if not diff_indices:
            return 0
            
        memo = {}
        
        def solve(indices_tuple):
            if not indices_tuple:
                return 0
            if indices_tuple in memo:
                return memo[indices_tuple]
            
            first_index = indices_tuple[0]
            min_cost = float('inf')
            
            for i in range(1, len(indices_tuple)):
                second_index = indices_tuple[i]
                pair_cost = min(x, second_index - first_index)
                remaining_indices_list = []
                for j in range(1, len(indices_tuple)):
                    if j != i:
                        remaining_indices_list.append(indices_tuple[j])
                remaining_indices_tuple = tuple(remaining_indices_list)
                recursive_cost = solve(remaining_indices_tuple)
                current_cost = pair_cost + recursive_cost
                min_cost = min(min_cost, current_cost)
                
            memo[indices_tuple] = min_cost
            return min_cost
            
        initial_indices_tuple = tuple(diff_indices)
        result = solve(initial_indices_tuple)
        return result