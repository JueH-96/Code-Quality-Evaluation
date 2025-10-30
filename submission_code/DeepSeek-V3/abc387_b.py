Wait, but the above approach may not be efficient for large n (like n=20), because the helper functions are recursive and may take exponential time. For example, count_alternating_permutations(n) is O(n!) in the worst case, which is not feasible for n=20.

Thus, the initial approach is not suitable for large n. We need a more efficient way to compute the counts or find the k-th permutation without enumerating all possibilities.

### Revised Approach
1. **Dynamic Programming for Counts**: We can use dynamic programming to precompute the number of valid permutations starting with each possible number. The idea is to memoize the counts for subsets of numbers with specific parity constraints.
2. **Constructing the Permutation Step by Step**: For each position in the permutation, we determine the next number by checking how many valid permutations start with each candidate number in lexicographical order. If the count of permutations starting with a candidate number is less than k, we subtract that count from k and move to the next candidate. Otherwise, we choose that candidate and proceed to the next position with the remaining numbers and updated parity constraint.

### Solution Code