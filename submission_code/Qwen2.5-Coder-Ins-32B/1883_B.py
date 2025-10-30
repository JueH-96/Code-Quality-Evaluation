# YOUR CODE HERE
import sys
from collections import Counter

def can_form_palindrome_with_k_removals(n, k, s):
    count = Counter(s)
    odd_count = sum(1 for c in count if count[c] % 2 != 0)
    
    # To form a palindrome, at most one character can have an odd count
    # We need to remove enough characters to make the number of odd counts <= 1
    # We can remove (odd_count - 1) / 2 pairs of characters to make all counts even
    # Plus, if odd_count is greater than 1, we can remove one more character to make it exactly 1 odd count
    # So, we need to remove at least (odd_count - 1) characters
    if odd_count - 1 <= k <= n:
        return "YES"
    return "NO"

def main():
    import sys
    input = sys.stdin.read().splitlines()
    index = 0
    t = int(input[index])
    index += 1
    results = []
    for _ in range(t):
        n, k = map(int, input[index].split())
        s = input[index + 1]
        index += 2
        results.append(can_form_palindrome_with_k_removals(n, k, s))
    print("
".join(results))

if __name__ == "__main__":
    main()