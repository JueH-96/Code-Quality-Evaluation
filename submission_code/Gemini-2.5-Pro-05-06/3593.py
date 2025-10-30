from typing import List

# Primes up to 30, as nums[i] <= 30. These are potential prime factors.
PRIMES_UP_TO_30 = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

class Solution:
    
    # Helper method to get the prime factorization of a number n (1 <= n <= 30).
    # Returns a dictionary mapping each prime in PRIMES_UP_TO_30 to its exponent in n.
    def _get_prime_factorization(self, n: int) -> dict:
        # Initialize factorization map with all primes from PRIMES_UP_TO_30,
        # with exponents starting at 0. This ensures consistency in keys across factorizations.
        factorization = {p: 0 for p in PRIMES_UP_TO_30}
        
        temp_n = n
        for p in PRIMES_UP_TO_30:
            # Optimization: if temp_n becomes 1, all its prime factors have been found.
            if temp_n == 1:
                break
            
            # Count occurrences of prime p as a factor of temp_n
            while temp_n > 0 and temp_n % p == 0: # temp_n > 0 check is mainly for safety, given n >= 1.
                factorization[p] += 1
                temp_n //= p
        
        # For n <= 30, temp_n must be 1 at this point, as all its prime factors
        # are less than or equal to 29 (the largest prime in PRIMES_UP_TO_30).
        return factorization

    # Helper method to calculate the factor score from a list of prime factorizations.
    # The factor score is defined as GCD * LCM.
    # This can be calculated using the property:
    # GCD * LCM = Product over all primes p of (p ^ (min_exponent_p + max_exponent_p)),
    # where min_exponent_p is the minimum exponent of p across all numbers in the list,
    # and max_exponent_p is the maximum.
    def _calculate_score_from_factorizations(self, factorizations_list: List[dict]) -> int:
        if not factorizations_list:
            # As per problem note: factor score of an empty array is 0.
            return 0
        
        if len(factorizations_list) == 1:
            # As per problem note: LCM and GCD of a single number are the number itself.
            # So, factor score = number * number.
            # We calculate this using its prime factorization:
            # If number = p1^e1 * p2^e2 * ..., then number^2 = p1^(2*e1) * p2^(2*e2) * ...
            score = 1
            single_num_factorization = factorizations_list[0]
            for p in PRIMES_UP_TO_30:
                exponent = single_num_factorization[p]
                # (p ** (2 * exponent)) correctly computes p^0=1 if exponent is 0.
                score *= (p ** (2 * exponent))
            return score

        # Case for multiple numbers in the list (i.e., len(factorizations_list) > 1)
        overall_score_product = 1
        for p in PRIMES_UP_TO_30:
            min_exponent_p = float('inf') # Smallest exponent of p among all numbers' factorizations
            max_exponent_p = 0            # Largest exponent of p among all numbers' factorizations
            
            for num_factorization in factorizations_list:
                min_exponent_p = min(min_exponent_p, num_factorization[p])
                max_exponent_p = max(max_exponent_p, num_factorization[p])
            
            # The contribution of this prime p to the total factor score product is
            # p ^ (min_exponent_p + max_exponent_p).
            # If prime p is not a factor in any number in the list, then its
            # min_exponent_p and max_exponent_p will both be 0.
            # In this case, p^(0+0) = p^0 = 1, so it correctly doesn't change the product.
            exponent_sum = min_exponent_p + max_exponent_p
            overall_score_product *= (p ** exponent_sum)
            
        return overall_score_product

    def maxScore(self, nums: List[int]) -> int:
        # Constraints: 1 <= nums.length <= 100, 1 <= nums[i] <= 30.

        # Precompute prime factorizations for all numbers in the input list 'nums'.
        # This is done once at the beginning to avoid redundant computations later.
        all_nums_factorizations = []
        for num_val in nums:
            all_nums_factorizations.append(self._get_prime_factorization(num_val))
            
        max_overall_factor_score = 0
        
        # Scenario 1: Calculate factor score without removing any element.
        # The list of factorizations for this scenario is 'all_nums_factorizations'.
        # The helper '_calculate_score_from_factorizations' correctly handles cases where 'nums'
        # (and thus 'all_nums_factorizations') might be empty (not per constraints, but robust),
        # contain one element, or contain multiple elements.
        score_no_removal = self._calculate_score_from_factorizations(all_nums_factorizations)
        max_overall_factor_score = score_no_removal # Initialize with the score of the original array.
        
        # Scenario 2: Calculate factor score after removing exactly one element.
        # Iterate through each element. For each, form a temporary sublist by removing that
        # element's factorization, then calculate the factor score for this sublist.
        num_elements_original = len(nums)
        
        # This loop iterates 'num_elements_original' times.
        # If 'num_elements_original' is 1 (e.g., nums=[x]):
        #   - 'score_no_removal' would be x*x. 'max_overall_factor_score' is initialized to x*x.
        #   - The loop runs once (for i=0).
        #   - 'temp_factorizations_list' becomes an empty list [].
        #   - '_calculate_score_from_factorizations([])' returns 0.
        #   - 'max_overall_factor_score' becomes max(x*x, 0), which is x*x. This logic is correct.
        for i in range(num_elements_original):
            # Create a temporary list of factorizations by excluding the i-th element's factorization.
            # This simulates removing the i-th number from 'nums'.
            temp_factorizations_list = all_nums_factorizations[:i] + all_nums_factorizations[i+1:]
            
            current_score_one_removal = self._calculate_score_from_factorizations(temp_factorizations_list)
            
            # Update the maximum score found so far.
            if current_score_one_removal > max_overall_factor_score:
                max_overall_factor_score = current_score_one_removal
                
        return max_overall_factor_score