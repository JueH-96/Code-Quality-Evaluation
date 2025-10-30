# YOUR CODE HERE
def max_11_22_substring(s):
    n = len(s)
    max_length = 1  # Minimum length is 1 (a single '/')
    
    for i in range(n):
        if s[i] == '/':
            left = right = 0
            
            # Count '1's to the left
            j = i - 1
            while j >= 0 and s[j] == '1':
                left += 1
                j -= 1
            
            # Count '2's to the right
            j = i + 1
            while j < n and s[j] == '2':
                right += 1
                j += 1
            
            # Calculate the length of the 11/22 substring
            current_length = min(left, right) * 2 + 1
            max_length = max(max_length, current_length)
    
    return max_length

# Read input
N = int(input())
S = input().strip()

# Solve and print the result
result = max_11_22_substring(S)
print(result)