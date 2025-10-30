class Solution:
    def numberOfPermutations(self, n: int, requirements: List[List[int]]) -> int:
        reqs_by_end_index = {}
        for end_index, count in requirements:
            reqs_by_end_index[end_index] = count
        
        dp = {} # dp[length][inversion_count][used_mask] = count
        dp[0] = {}
        dp[0][0] = {}
        dp[0][0][0] = 1
        
        for length in range(1, n + 1):
            dp[length] = {}
            for mask in range(1 << n):
                if bin(mask).count('1') != length:
                    continue
                dp[length][0] = dp[length].get(0, {})
                dp[length][0][mask] = 0
                for inv_count in range(401): # Maximum possible inversion count constraint is 400
                    dp[length][inv_count] = dp[length].get(inv_count, {})
                    dp[length][inv_count][mask] = 0
                    
                for num_index in range(n):
                    if (mask >> num_index) & 1: # if num_index is in the current mask
                        prev_mask = mask ^ (1 << num_index)
                        current_number = num_index
                        inversions_created = 0
                        for prev_num_index in range(n):
                            if (prev_mask >> prev_num_index) & 1:
                                if prev_num_index > current_number:
                                    inversions_created += 1
                                    
                        for prev_inv_count in range(401):
                            if prev_inv_count in dp[length-1] and prev_mask in dp[length-1][prev_inv_count]:
                                count = dp[length-1][prev_inv_count][prev_mask]
                                if count > 0:
                                    current_inv_count = prev_inv_count + inversions_created
                                    if current_inv_count <= 400:
                                        required_inv_count = -1
                                        if length - 1 in reqs_by_end_index:
                                            required_inv_count = reqs_by_end_index[length - 1]
                                        if required_inv_count == -1 or current_inv_count == required_inv_count:
                                            dp[length][current_inv_count][mask] = dp[length][current_inv_count].get(mask, 0) + count
                                            
        final_inversion_count = -1
        if n - 1 in reqs_by_end_index:
            final_inversion_count = reqs_by_end_index[n - 1]
        else:
            return 0 # Should not happen based on problem description
            
        if final_inversion_count == -1:
            total_permutations = 0
            for inv_count in dp[n]:
                total_permutations = (total_permutations + sum(dp[n][inv_count].values())) % (10**9 + 7)
            return total_permutations
        else:
            if final_inversion_count in dp[n] and ((1 << n) - 1) in dp[n][final_inversion_count]:
                return dp[n][final_inversion_count][(1 << n) - 1] % (10**9 + 7)
            else:
                return 0