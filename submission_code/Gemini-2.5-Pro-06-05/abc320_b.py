# YOUR CODE HERE
import sys

def solve():
    """
    Reads a string from standard input and finds the maximum length
    of a contiguous palindromic substring.
    """
    S = sys.stdin.readline().strip()
    n = len(S)
    
    # The minimum possible answer is 1, as any single character is a palindrome.
    max_len = 1
    
    # Generate all contiguous substrings
    # The outer loop sets the starting point of the substring
    for i in range(n):
        # The inner loop sets the ending point of the substring
        for j in range(i, n):
            # Extract the substring S[i...j]
            substring = S[i:j+1]
            
            # Check if the substring is a palindrome by comparing it to its reverse
            if substring == substring[::-1]:
                # If it's a palindrome, update the maximum length found so far
                current_len = len(substring)
                if current_len > max_len:
                    max_len = current_len
                    
    print(max_len)

solve()