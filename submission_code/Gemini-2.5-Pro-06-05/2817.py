class Solution:
    def minimumCost(self, s: str) -> int:
        """
        Calculates the minimum cost to make all characters in a binary string equal.

        The core idea is that to make a string uniform (all '0's or all '1's),
        we must ensure that all adjacent characters are equal (s[i] == s[i+1]).

        Consider a mismatch where s[i] != s[i+1]. To resolve this, we must
        apply an operation that flips one character but not the other. The only
        operations that achieve this are those whose boundary is between i and i+1:
        1. Flip the prefix s[0...i]: This has a cost of i + 1.
        2. Flip the suffix s[i+1...n-1]: This has a cost of n - (i + 1).

        To fix this specific mismatch with minimum cost, we choose the cheaper of
        these two options. The cost is min(i + 1, n - (i + 1)).

        Crucially, fixing a mismatch at index `i` does not resolve any other
        mismatch at a different index `j`. The operations either leave the `j, j+1`
        pair untouched or flip both, preserving their inequality. This means the
        cost to fix each mismatch is independent.

        Therefore, the total minimum cost is the sum of the minimum costs to resolve
        each individual mismatch found in the string.
        """
        n = len(s)
        
        # A string of length 1 or less is already uniform.
        if n <= 1:
            return 0
            
        total_cost = 0
        
        # Iterate through all adjacent pairs to find mismatches.
        for i in range(n - 1):
            if s[i] != s[i+1]:
                # Cost to flip prefix s[0...i]
                cost_prefix_flip = i + 1
                
                # Cost to flip suffix s[i+1...n-1]
                cost_suffix_flip = n - (i + 1)
                
                # Add the minimum of the two costs to the total.
                total_cost += min(cost_prefix_flip, cost_suffix_flip)
                
        return total_cost