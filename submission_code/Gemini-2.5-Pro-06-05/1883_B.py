import sys
from collections import Counter

def solve():
    """
    Solves a single test case for the palindrome removal problem.
    """
    try:
        # Read n and k from the first line of the test case
        line1 = sys.stdin.readline()
        if not line1:
            return
        n, k = map(int, line1.split())
        
        # Read the string s from the second line
        s = sys.stdin.readline().strip()

    except (IOError, ValueError):
        return

    # A string can be rearranged into a palindrome if at most one character
    # type appears an odd number of times.
    
    # Count the frequency of each character in the string.
    counts = Counter(s)

    # Count how many character types have an odd frequency.
    odd_counts = 0
    for count in counts.values():
        if count % 2 != 0:
            odd_counts += 1

    # To make the string palindromable, we need to reduce the number of
    # odd-frequency character types to at most 1.
    # To reduce the number of odd types from `odd_counts` to 1, we need to
    # "fix" `odd_counts - 1` of them by changing their parity from odd to even.
    # Each fix requires at least one removal of a character of that type.
    # Therefore, we need at least `odd_counts - 1` removals.
    
    # If the number of available removals `k` is less than `odd_counts - 1`,
    # it's impossible to achieve the goal.
    # Let's check if this necessary condition is also sufficient.
    # If `k >= odd_counts - 1`, we can use `odd_counts - 1` removals to leave
    # just one odd-frequency type. The remaining `k - (odd_counts - 1)` removals
    # can be made without breaking the palindromic property.
    # If `k - (odd_counts - 1)` is even, we remove pairs of characters.
    # If `k - (odd_counts - 1)` is odd, we remove the last odd-frequency character
    # and then remove pairs.
    # Since k < n, we always have enough characters to perform these removals.
    # Thus, the condition `k >= odd_counts - 1` is both necessary and sufficient.
    
    if k < odd_counts - 1:
        print("NO")
    else:
        print("YES")

def main():
    """
    Main function to handle multiple test cases.
    """
    try:
        # Read the number of test cases
        num_test_cases_str = sys.stdin.readline()
        if not num_test_cases_str:
            return
        num_test_cases = int(num_test_cases_str)
        
        for _ in range(num_test_cases):
            solve()
            
    except (IOError, ValueError):
        # Graceful exit on empty input or invalid format
        return

if __name__ == "__main__":
    main()