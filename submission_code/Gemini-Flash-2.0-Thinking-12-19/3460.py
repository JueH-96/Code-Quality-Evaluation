class Solution:
    def numberOfPermutations(self, n: int, requirements: List[List[int]]) -> int:
        req_map = {}
        for end_index, count in requirements:
            req_map[end_index] = count
        memo = {}
        mod = 10**9 + 7
        
        def solve(index, current_inversions, used_mask):
            if index == n:
                return 1
            if (index, current_inversions, used_mask) in memo:
                return memo[(index, current_inversions, used_mask)]
            
            end_index_to_check = index - 1
            if end_index_to_check in req_map:
                required_inversions = req_map[end_index_to_check]
                if current_inversions != required_inversions:
                    return 0
                    
            count = 0
            for num in range(n):
                if not (used_mask & (1 << num)):
                    new_inversions_count = 0
                    for prev_num in range(n):
                        if (used_mask & (1 << prev_num)) and prev_num > num:
                            new_inversions_count += 1
                    next_inversions = current_inversions + new_inversions_count
                    next_used_mask = used_mask | (1 << num)
                    count = (count + solve(index + 1, next_inversions, next_used_mask)) % mod
                    
            memo[(index, current_inversions, used_mask)] = count
            return count
            
        return solve(0, 0, 0)