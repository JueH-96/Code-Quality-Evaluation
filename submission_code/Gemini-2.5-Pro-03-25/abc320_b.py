import sys

def solve():
    s = sys.stdin.readline().strip()
    n = len(s)
    max_len = 0

    # Handle the edge case where the string itself might be empty or very short,
    # although constraints say length >= 2.
    # Every single character is a palindrome of length 1.
    if n > 0:
        max_len = 1
    else:
        print(0) # Should not happen based on constraints
        return

    # Iterate through all possible contiguous substrings
    for i in range(n):
        for j in range(i, n):
            # Extract the substring
            substring = s[i : j + 1]

            # Check if the substring is a palindrome
            # A string is a palindrome if it reads the same forwards and backwards.
            # We can check this by comparing the string with its reverse.
            if substring == substring[::-1]:
                # If it's a palindrome, update the maximum length found so far
                current_len = len(substring)
                if current_len > max_len:
                    max_len = current_len

    print(max_len)

# Read input and call the solve function
solve()