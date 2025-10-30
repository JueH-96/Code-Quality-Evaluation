import sys
import collections

# Global memoization dictionary to store results of solve(N)
memo = {}

# Precompute numbers consisting only of digits 1-9 up to a certain limit.
# This limit is chosen to balance between search space and time.
# For N <= 10^12, max factor can be sqrt(N) approx 10^6.
# So, we generate numbers up to 1,000,000 (10^6), which have at most 6 digits.
# There are 9^6 = 531,441 such numbers. This is a feasible number of iterations per DFS call.
MAX_FACTOR_VAL = 1000000 
NON_ZERO_DIGIT_NUMBERS = []
q = collections.deque([''])

# BFS to generate numbers with no '0' digits.
# Start with an empty string, append digits 1-9.
while q:
    s = q.popleft()
    if s != '':
        num = int(s)
        if num > MAX_FACTOR_VAL:
            # If current number exceeds limit, any further numbers built from it
            # (by appending more digits) will also exceed the limit.
            continue
        NON_ZERO_DIGIT_NUMBERS.append(num)
    
    # Pruning the BFS tree: Only append new digits if the current string length
    # is less than the number of digits in MAX_FACTOR_VAL.
    # For MAX_FACTOR_VAL = 1,000,000, len(str(MAX_FACTOR_VAL)) is 7.
    # So, we append if len(s) < 7, meaning we generate strings up to 6 digits long.
    if len(s) < len(str(MAX_FACTOR_VAL)):
        for d_char in '123456789':
            q.append(s + d_char)

# Sort the numbers to try smaller factors first (might lead to different valid solutions).
NON_ZERO_DIGIT_NUMBERS.sort()


def solve(current_N):
    """
    Recursively finds a list of string factors for current_N such that their product
    equals current_N and their concatenated string (with '*' separators) is a palindrome.
    
    Args:
        current_N (int): The target number to factor.
        
    Returns:
        list[str] or None: A list of string factors if a solution is found, otherwise None.
    """
    if current_N in memo:
        return memo[current_N]

    # Base case 1: current_N itself forms a valid single-factor string.
    # Check if it contains '0' and if it's a palindrome.
    s_N = str(current_N)
    if '0' not in s_N and s_N == s_N[::-1]:
        # Check string length constraint (should be satisfied for N <= 10^12, max 13 digits)
        if len(s_N) <= 1000:
            memo[current_N] = [s_N]
            return [s_N]

    # Base case 2: If current_N is 1, it means we have successfully factored all required numbers.
    # This signifies an empty sequence of factors, resulting in a product of 1.
    if current_N == 1:
        memo[current_N] = []
        return []

    # Recursive step: try to find two outer factors f_left and f_right.
    # f_left is a number, and f_right is its string reverse converted to an integer.
    # The product f_left * f_right must divide current_N.
    for f_left in NON_ZERO_DIGIT_NUMBERS:
        # Optimization: if f_left is already greater than current_N,
        # then f_left * f_right will certainly be greater than current_N (since f_right >= 1).
        # As NON_ZERO_DIGIT_NUMBERS is sorted, we can break.
        if f_left > current_N:
             break

        # Check if f_left is a divisor of current_N.
        if current_N % f_left != 0:
            continue

        s_left = str(f_left)
        s_right = s_left[::-1] # Get the string reverse of f_left
        f_right = int(s_right) # Convert the reversed string back to an integer

        # Since f_left consists only of non-zero digits, s_right also consists only of non-zero digits.
        # Thus, f_right will always be > 0.
        
        # Check for potential integer overflow or division by zero (not an issue in Python for these values).
        # More importantly, check if f_left * f_right exceeds current_N.
        # This is equivalent to checking f_left > current_N // f_right.
        if f_right == 0 or f_left > current_N // f_right:
             continue
        
        # Check if f_left * f_right evenly divides current_N.
        if current_N % (f_left * f_right) == 0:
            remaining_N = current_N // (f_left * f_right)
            
            # Recursive call to find the inner part of the palindrome.
            sub_res = solve(remaining_N)

            # If a solution was found for the inner part:
            if sub_res is not None:
                # Construct the full candidate string parts: [s_left] + sub_res + [s_right].
                # If sub_res is empty ([]), it means remaining_N was 1, and candidate_parts becomes [s_left, s_right].
                candidate_parts = [s_left] + sub_res + [s_right]
                
                # Join with '*' to form the full potential solution string.
                full_string = "*".join(candidate_parts)

                # Final checks: string length constraint and overall palindrome property.
                if len(full_string) <= 1000 and full_string == full_string[::-1]:
                    memo[current_N] = candidate_parts
                    return candidate_parts

    # If no solution found after trying all possible f_left values, memoize None.
    memo[current_N] = None
    return None

# Main part of the script
if __name__ == '__main__':
    # Increase the recursion limit for deep DFS calls, as max depth can be around 500.
    sys.setrecursionlimit(1000) 

    N = int(sys.stdin.readline().strip())

    result_parts = solve(N)

    if result_parts is None:
        sys.stdout.write("-1
")
    else:
        sys.stdout.write("*".join(result_parts) + "
")