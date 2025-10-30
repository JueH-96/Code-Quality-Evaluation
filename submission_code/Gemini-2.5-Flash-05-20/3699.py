class Solution:
    def numberOfSubsequences(self, nums: List[int]) -> int:
        N = len(nums)
        max_num = 1000
        max_product_val = max_num * max_num # Maximum possible product for nums[q] * nums[s] (1000*1000 = 10^6)

        ans = 0

        # Precompute prefix_counts
        # prefix_counts[i][val] = count of 'val' in nums[0...i]
        # Size: N rows, (max_num + 1) columns
        # Index i ranges from 0 to N-1
        prefix_counts = [[0] * (max_num + 1) for _ in range(N)]
        prefix_counts[0][nums[0]] = 1 # Initialize for the first element
        for i in range(1, N):
            # Copy counts from previous prefix
            for val in range(1, max_num + 1):
                prefix_counts[i][val] = prefix_counts[i-1][val]
            # Increment count for the current number
            prefix_counts[i][nums[i]] += 1
        
        # Precompute divisors for all possible product values up to max_product_val
        # all_divs[k] stores a list of divisors of k
        # Size: (max_product_val + 1) lists
        all_divs = [[] for _ in range(max_product_val + 1)]
        for i in range(1, max_product_val + 1):
            # For each number i, iterate its multiples j
            # i is a divisor of j
            for j in range(i, max_product_val + 1, i):
                all_divs[j].append(i)

        # Helper function to get count of a value in a specific range [start_idx, end_idx]
        # This uses the precomputed prefix_counts
        def get_range_count(start_idx, end_idx, val):
            # If the range is invalid (start index after end index), return 0
            if start_idx > end_idx:
                return 0
            
            # Count up to end_idx
            count = prefix_counts[end_idx][val]
            # Subtract counts before start_idx if start_idx is not 0
            if start_idx > 0:
                count -= prefix_counts[start_idx - 1][val]
            return count

        # Iterate q (second index in p < q < r < s)
        # q must be at least 2 (p=0, q=2)
        # q must be at most N-5 (to allow r=q+2, s=r+2=q+4, up to N-1 for s)
        for q in range(2, N - 4): 
            # Iterate s (fourth index in p < q < r < s)
            # s must be at least q+4 (p=0, q=2, r=4, s=6)
            # s must be at most N-1
            for s in range(q + 4, N): 
                # The condition is nums[p] * nums[r] == nums[q] * nums[s]
                # Fix nums[q] and nums[s], calculate their product (RHS of the equation)
                target_product = nums[q] * nums[s]

                # We need to find pairs (val_p, val_r) such that val_p * val_r == target_product
                # val_p is the value nums[p], val_r is the value nums[r]
                # Iterate through all possible values for val_p (which are divisors of target_product)
                for val_p in all_divs[target_product]:
                    # val_p (nums[p]) must be within the allowed range [1, max_num]
                    # Since nums[i] are positive, val_p >= 1 is implicit from all_divs generation.
                    if val_p > max_num: 
                        continue 

                    # Calculate the corresponding value for nums[r] (val_r)
                    val_r = target_product // val_p
                    
                    # val_r (nums[r]) must also be within the allowed range [1, max_num]
                    if val_r > max_num: 
                        continue

                    # Count valid p's: index p must be in range [0, q-2]
                    # This ensures p < q and q - p > 1
                    count_p = get_range_count(0, q - 2, val_p)

                    # Count valid r's: index r must be in range [q+2, s-2]
                    # This ensures q < r, r - q > 1, r < s, and s - r > 1
                    count_r = get_range_count(q + 2, s - 2, val_r)
                    
                    # Add the product of counts to the total answer
                    # Each combination of a valid p and a valid r contributes one special subsequence
                    ans += count_p * count_r
        
        return ans