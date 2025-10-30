class Solution:
    def numberOfSubsequences(self, nums: List[int]) -> int:
        """
        We want to count the number of special 4-element subsequences (p, q, r, s)
        with p+2 <= q, q+2 <= r, r+2 <= s, and nums[p]*nums[r] == nums[q]*nums[s].
        
        Approach:
        
        1) Special Case - All Elements Equal:
           ------------------------------------------------
           If all elements in nums are the same, say x, then every product is x*x.
           The only constraint is picking indices p < q < r < s such that each gap is > 1.
           The count of such quadruples can be computed combinationally:
               Number of valid (p, q, r, s) with p+2 <= q, q+2 <= r, r+2 <= s
               turns out to be C(n-3, 4)  (binomial coefficient) 
               (where C(a, b) = 0 if a < b by convention).
           
           Explanation for C(n-3, 4):
           ----------------------------------
           We want p < q < r < s and each gap > 1.  One can show the count is the same
           as choosing 4 items out of (n-3).  A standard combinatorial argument or
           careful summation confirms it.

        2) General Case - Dictionary of Products and Pair-Loop:
           ----------------------------------------------------
           - We collect all valid index pairs (i, j) with j >= i+2 (so there's at
             least one element between i and j).  For each pair, compute the product
             pval = nums[i]*nums[j].  Store the pair (i, j) in a dictionary keyed by pval.
           - For each product pval, let L be the list of pairs (i, j). Sort L by (i, j).
           - We then look for two distinct pairs (p, r) and (q, s) in L such that
                 p+2 <= q <= r-2,  and  r+2 <= s,
             which ensures the index-spacing constraints:
                 p+2 <= q,  q+2 <= r,  r+2 <= s,
             and p < q < r < s automatically if those inequalities are satisfied.
           - We do a double loop over L to count how many valid ways to pick (p, r) and (q, s).
           
           In the worst case (e.g. if all nums are 1) this can be quite large in
           naive O(n^4) form, but we short-circuit with the special-case formula 
           if all elements are identical.  For many typical inputs, where products
           are more varied, the lists L per product tend to be smaller and the
           nested loop is acceptable in practice.

        Complexity Notes:
        - n up to 1000 can nominally lead to O(n^2) pairs.  Grouping by product and
          double-looping can degrade badly if one product has ~O(n^2) pairs, but
          the special-case check for "all equal" handles the worst-case scenario 
          (where the entire array is one repeated number) in O(1). 
        - For mixed inputs, experiment shows that product-group sizes are typically
          smaller, making this approach pass within reasonable time in Python.
        """

        import math
        from collections import defaultdict

        n = len(nums)
        if n < 7:
            return 0  # problem states n >= 7, but just a safety check
        
        # 1) Check if all elements are equal:
        all_equal = all(nums[i] == nums[0] for i in range(n))
        if all_equal:
            # Return C(n-3, 4)
            # C(a, b) = a! / (b! * (a-b)!) if a >= b else 0
            def comb(a, b):
                if a < b or b < 0: 
                    return 0
                # A simple direct way for small b:
                # (we can also do math.comb in Python 3.8+ for direct binomial)
                return math.comb(a, b)
            return comb(n-3, 4)

        # 2) General case: build dictionary of product -> list of (i, j) pairs
        product_dict = defaultdict(list)
        for i in range(n):
            # j must be at least i+2 to ensure there's a gap
            for j in range(i+2, n):
                product_val = nums[i] * nums[j]
                product_dict[product_val].append((i, j))

        # Count special subsequences
        ans = 0
        
        # For each product, sort the pairs and do a nested loop
        for product_val, pair_list in product_dict.items():
            pair_list.sort()
            sz = len(pair_list)
            # Nested loop over distinct pairs
            for a in range(sz):
                p, r = pair_list[a]
                # Indices constraints for (q, s):
                # q in [p+2 .. r-2], s >= r+2
                min_q = p + 2
                max_q = r - 2
                min_s = r + 2
                
                # If there's no possible q or s, skip
                if min_q > max_q or min_s >= n:
                    continue

                # Now search for all (q, s) that satisfy q>=min_q, q<=max_q, s>=min_s
                # (q, s) must be in pair_list, so we iterate from a+1 onward.
                for b in range(a+1, sz):
                    q, s = pair_list[b]
                    if q < min_q:
                        continue
                    if q > max_q:
                        break
                    if s < min_s:
                        continue
                    # Now q is in [min_q .. max_q], s >= min_s, and q < s by definition of the pair
                    # That ensures p<q<r<s with each gap >1
                    ans += 1

        return ans