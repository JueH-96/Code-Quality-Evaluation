def can_form_palindrome(s):
    # Count frequencies of each character
    freq = {}
    for c in s:
        freq[c] = freq.get(c, 0) + 1
    
    # Count odd frequencies
    odd_count = sum(1 for f in freq.values() if f % 2 == 1)
    
    # For palindrome, at most 1 character can have odd frequency
    return odd_count <= 1

def solve_test_case(n, k, s):
    # Get frequency of each character
    freq = {}
    for c in s:
        freq[c] = freq.get(c, 0) + 1
    
    # After removing k characters, n-k characters remain
    remaining = n - k
    
    # If remaining length is 0 or 1, it's always possible
    if remaining <= 1:
        return "YES"
    
    # Try removing k characters
    # For each character count, we can remove 0 to min(k, count) characters
    # Use dynamic programming to try all possibilities
    possible_counts = {0: 0}  # key: number of odd frequencies, value: chars removed
    
    for c, count in freq.items():
        new_possible = {}
        for curr_odd, chars_removed in possible_counts.items():
            # Don't remove any
            new_odd = curr_odd
            if count % 2 == 1:
                new_odd = curr_odd + 1
            if chars_removed <= k:
                new_possible[new_odd] = min(new_possible.get(new_odd, float('inf')), chars_removed)
            
            # Remove 1 to count characters
            for remove in range(1, count + 1):
                new_count = count - remove
                new_odd = curr_odd
                if new_count % 2 == 1:
                    new_odd = curr_odd + 1
                if chars_removed + remove <= k:
                    new_possible[new_odd] = min(new_possible.get(new_odd, float('inf')), chars_removed + remove)
        
        possible_counts = new_possible
    
    # Check if any possibility uses exactly k removals and results in valid palindrome
    for odd_count, chars_removed in possible_counts.items():
        if chars_removed == k and odd_count <= 1:
            return "YES"
    
    return "NO"

# Read number of test cases
t = int(input())

# Process each test case
for _ in range(t):
    # Read n and k
    n, k = map(int, input().split())
    # Read string s
    s = input().strip()
    # Print result
    print(solve_test_case(n, k, s))