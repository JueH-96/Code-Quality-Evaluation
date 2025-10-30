from collections import Counter

def can_form_palindrome(s, k):
    # Count the frequency of each character in the string
    freq = Counter(s)
    # Count the number of characters with odd frequency
    odd_count = sum(1 for count in freq.values() if count % 2 != 0)
    # We can remove pairs of characters to make them even, and we can leave at most one character with an odd count
    # If the number of characters we need to remove (k) is greater than or equal to the number of odd characters minus one,
    # then it is possible to form a palindrome
    return k >= odd_count - (len(s) % 2)

# Read the number of test cases
t = int(input().strip())
for _ in range(t):
    # Read the values of n and k for each test case
    n, k = map(int, input().strip().split())
    # Read the string s
    s = input().strip()
    # Check if it is possible to form a palindrome and print the result
    print("YES" if can_form_palindrome(s, k) else "NO")