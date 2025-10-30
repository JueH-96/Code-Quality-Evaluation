# YOUR CODE HERE
import sys
from collections import Counter

def solve():
    """
    Solves a single test case for the Yarik and Music problem.
    """
    try:
        line = sys.stdin.readline()
        if not line: return
        n = int(line)
        
        a = list(map(int, sys.stdin.readline().split()))
    except (IOError, ValueError):
        return

    # The problem asks for the number of pairs (i, j) with i < j such that
    # b_i^(b_j) = b_j^(b_i), where b_k = 2^(a_k).
    #
    # Substituting b_i and b_j into the equation:
    # (2^(a_i))^(2^(a_j)) = (2^(a_j))^(2^(a_i))
    # 2^(a_i * 2^(a_j)) = 2^(a_j * 2^(a_i))
    #
    # Since the bases are equal, we can equate the exponents:
    # a_i * 2^(a_j) = a_j * 2^(a_i)
    #
    # For positive integers a_i and a_j (as per problem constraints 1 <= a_i),
    # this equation holds if and only if one of two conditions is met:
    # 1. a_i = a_j. In this case, the equation is trivially true.
    # 2. {a_i, a_j} = {1, 2}. We can verify this:
    #    Let a_i=1, a_j=2.  1 * 2^2 = 2 * 2^1  => 4 = 4. True.
    #    It can be shown that this is the only pair of distinct integers that works.
    #
    # These two conditions are mutually exclusive (in the second, a_i != a_j).
    # Therefore, we can count the pairs satisfying each condition and sum them up.

    counts = Counter(a)
    
    total_pairs = 0
    
    # Case 1: Count pairs where a_i = a_j.
    # If a number appears 'c' times in the array, it corresponds to 'c' different
    # indices. The number of pairs of indices we can form is "c choose 2",
    # which is c * (c - 1) / 2.
    for count in counts.values():
        total_pairs += count * (count - 1) // 2
            
    # Case 2: Count pairs where {a_i, a_j} = {1, 2}.
    # The number of such pairs of indices is the number of elements with value 1
    # multiplied by the number of elements with value 2.
    # We use counts.get(k, 0) to safely handle cases where 1 or 2 might not be present.
    total_pairs += counts.get(1, 0) * counts.get(2, 0)
    
    print(total_pairs)

def main():
    """
    Main function to handle multiple test cases.
    """
    try:
        num_tests_str = sys.stdin.readline()
        if not num_tests_str.strip():
            return
        num_tests = int(num_tests_str)
        for _ in range(num_tests):
            solve()
    except (IOError, ValueError):
        # Gracefully handle potential empty input or parsing errors
        return

if __name__ == "__main__":
    main()