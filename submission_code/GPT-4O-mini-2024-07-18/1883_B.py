def can_form_palindrome_after_removal(t, test_cases):
    results = []
    for n, k, s in test_cases:
        from collections import Counter
        
        # Count frequency of each character
        freq = Counter(s)
        
        # Count how many characters have an odd frequency
        odd_count = sum(1 for count in freq.values() if count % 2 != 0)
        
        # Calculate the number of characters that can remain after removing k characters
        remaining_length = n - k
        
        # To form a palindrome:
        # If remaining_length is even, we can have at most 0 odd counts
        # If remaining_length is odd, we can have at most 1 odd count
        if remaining_length % 2 == 0:
            # Even length can have 0 odd counts
            if odd_count <= 0:
                results.append("YES")
            else:
                results.append("NO")
        else:
            # Odd length can have at most 1 odd count
            if odd_count <= 1:
                results.append("YES")
            else:
                results.append("NO")
    
    return results

# Read input
import sys
input = sys.stdin.read
data = input().splitlines()

t = int(data[0])
test_cases = []

index = 1
for _ in range(t):
    n, k = map(int, data[index].split())
    s = data[index + 1]
    test_cases.append((n, k, s))
    index += 2

# Get results
results = can_form_palindrome_after_removal(t, test_cases)

# Print results
print("
".join(results))