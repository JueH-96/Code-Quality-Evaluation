Explanation:

1. The function pop_sum(n) computes the total number of “elements” (i.e. set bits) in the binary representations of 1 through n.
2. The function exp_sum(n) computes the total sum of exponents in these representations.
3. For a given integer n, partial_exponent(n, k) returns the sum of the exponents in the first k (lowest) set bits (note that n’s binary representation is already “sorted” if we read bits from lower to higher indices).
4. The helper find_n(pos) uses binary search (over n in [1, pos+1]) to find the integer whose representation covers the big_nums index pos.
5. Then prefix_E(pos) returns the sum of exponents for all elements in big_nums up to index pos.
6. Finally, each query’s answer is computed as pow(2, (prefix_E(to)-prefix_E(fr-1)), mod).

This solution meets the problem specification and passes the sample tests.