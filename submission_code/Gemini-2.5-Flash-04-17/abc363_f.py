import sys
import math
import functools

# Increase recursion depth limit for potentially deep searches
# The maximum number of terms is limited by string length ~1000,
# with minimal term length 1 and '*' separator length 1, roughly 500 terms.
# The recursion depth corresponds to the number of pairs added + the middle term.
# A deep recursion might happen if factors are small.
sys.setrecursionlimit(2000) # Increased to 2000, maybe sufficient

def rev_val(s):
    """Converts reversed string s to integer. Returns -1 if invalid."""
    reversed_s = s[::-1]
    # Check if reversed string is just "0" or starts with "0"
    if reversed_s == "0": # Although terms > 0, rev_val could theoretically produce 0 from "0"
        return -1
    if reversed_s.startswith('0'):
        return -1
    try:
        val = int(reversed_s)
        # Ensure the integer value also doesn't contain '0'
        return val if has_no_zero(val) else -1
    except ValueError:
        return -1

def has_no_zero(n):
    """Checks if integer n contains digit 0 in its string representation."""
    return '0' not in str(n)

def is_palindrome_string(s):
    """Checks if string s is a palindrome."""
    return s == s[::-1]

def get_divisors(n):
    """Returns a list of divisors of n, in increasing order."""
    divisors = []
    # Find divisors up to sqrt(n)
    limit = int(math.sqrt(n))
    for i in range(1, limit + 1):
        if n % i == 0:
            divisors.append(i)
            if i * i != n:
                divisors.append(n // i)
    # Sort to explore smaller first
    divisors.sort()
    return divisors

# Memoization dictionary
memo = {}

def solve(current_N):
    """
    Recursive function to find a list of term strings [S1, S2, ..., Sk]
    such that their product evaluates to current_N and the string
    S1*S2*...*Sk is a palindrome string consisting of digits 1-9 and *.

    Args:
        current_N: The integer product required from this sequence of terms.

    Returns:
        A list of strings [S1, S2, ..., Sk] if a solution is found,
        or None otherwise.
    """
    if current_N in memo:
        return memo[current_N]

    # Check if current_N can be a single term for the current subproblem
    # A single term 'T' must satisfy product = T, str(T) is palindrome, str(T) has no '0'.
    # This is the base case of the recursion: current_N is itself a palindrome term.
    if has_no_zero(current_N):
        mid_candidate_str = str(current_N)
        if is_palindrome_string(mid_candidate_str):
            # This is a valid sequence for current_N: just [str(current_N)]
            # We don't check length here; length check happens in the caller
            # when this sequence is combined with outer terms.
             memo[current_N] = [mid_candidate_str]
             return memo[current_N]

    # If current_N cannot be a single palindrome term,
    # it must be represented as a product of multiple terms [d, Mid..., rev_val(str(d))].
    # current_N = d * Product(Mid...) * rev_val(str(d))
    # The string "str(d) * Mid_string * str(rev_val(str(d)))" must be palindrome,
    # where Mid_string is "*".join(solve(Product(Mid...))).

    # Try finding outer terms [d, rev_val(str(d))] for the sequence that multiplies to current_N
    # Iterate through potential first terms 'd' of the sequence
    divisors = get_divisors(current_N)
    for d in divisors:
        # 'd' must be a term in the product, so str(d) must not contain '0'
        if not has_no_zero(d):
            continue

        S1 = str(d)
        S_rev = S1[::-1]
        P_rev_val = rev_val(S_rev) # Integer value of the reversed string

        # Check if the reversed string forms a valid integer term > 0
        # and if d * P_rev_val divides current_N
        if P_rev_val > 0 and current_N % (d * P_rev_val) == 0:
            next_N = current_N // (d * P_rev_val)

            # Recursively find the terms for the middle part (whose product is next_N)
            # The recursive call solve(next_N) should return a list of strings [Mid1, ..., Midj]
            # such that Mid1*...*Midj = next_N and '*'.join(Mid_i) is palindrome.
            middle_terms_result = solve(next_N)

            # If the recursive call found a solution for the middle part
            if middle_terms_result is not None:
                # Construct the full sequence of terms for current_N
                # This sequence is [d] + middle_terms + [P_rev_val]
                # The strings are [S1] + middle_terms_result + [S_rev]
                potential_terms = [S1] + middle_terms_result + [S_rev]

                # Calculate the length of the string representation
                potential_string = '*'.join(potential_terms)

                # Check length constraint *here* before accepting this path
                # This pruning is crucial for performance and correctness
                if len(potential_string) <= 1000:
                    # Found a valid sequence of terms for current_N within length limit
                    # Store and return the list of strings
                    memo[current_N] = potential_terms
                    return memo[current_N]
                # Else: This path leads to a string too long, continue searching other divisors/paths

    # If no solution found for current_N after trying all pairs
    memo[current_N] = None
    return None

# Read input
N = int(sys.stdin.readline())

# Check the single term case for N itself first
# This is the simplest possible solution. If it works, print it and exit.
# Ensure N is positive as per constraints
if N >= 1 and has_no_zero(N):
     single_term_str = str(N)
     if is_palindrome_string(single_term_str) and len(single_term_str) <= 1000:
         # Verify product just in case (should be N)
         try:
             # int(single_term_str) == N is guaranteed since single_term_str = str(N)
             print(single_term_str)
             sys.exit()
         except ValueError: # Should not happen if has_no_zero check passes
             pass # Fall through to recursive search if single term check has issue


# If the single term case didn't work, start the recursive search
# This search finds a list of strings [S1, ..., Sk] for the entire N.
result_terms_strings = solve(N)

# Check the result from the recursive search
if result_terms_strings is not None:
    # If solve(N) returned a list (non-empty or empty), it represents a sequence for N.
    # An empty list [] from solve(N) implies N=1 and no terms needed for the middle (which was N itself).
    # This [] case would happen only if N=1 and solve(1) returns [].
    # But solve(1) returns ["1"] if N=1.
    # So, if result_terms_strings is not None, it must be a list of strings [S1, ..., Sk] where k >= 1.
    # Because the base case returns a list with >= 1 element if current_N >= 1,
    # and recursive step combines lists, the resulting list should not be empty if N >= 1.

    if result_terms_strings: # Check if the list is not empty (k >= 1)
        final_string = '*'.join(result_terms_strings)

        # Final verification (should pass based on solve logic)
        # Length <= 1000 checked in solve before returning results up.
        # Palindrome checked implicitly by recursive structure and base case.
        # No '0' checked by has_no_zero and rev_val.
        # First char digit checked by iterating d from 1 and has_no_zero.
        # Value product checked by current_N division.

        # Assuming logic is correct, print the found string
        # The necessary checks (palindrome, length, no '0', first digit) are implicitly handled
        # by the structure of the recursion and base case, and explicit checks within solve.
        print(final_string)

    else: # result_terms_strings is [] (empty list) - Should not happen for N >= 1
        # This case implies solve(N) returned [], which would only happen if solve(1) returned [].
        # solve(1) returns ["1"], so this branch should not be reached for N >= 1.
        sys.stderr.write(f"Error: solve({N}) returned empty list.
") # Debugging output
        print("-1") # Fallback print -1


else: # result_terms_strings is None
    # No solution found by recursive search
    print("-1")