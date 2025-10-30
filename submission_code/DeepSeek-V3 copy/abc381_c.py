# YOUR CODE HERE
def max_11_22_substring_length(N, S):
    max_len = 0
    # Iterate through each character in the string
    for i in range(N):
        if S[i] == '/':
            # Initialize the left and right pointers
            left = i - 1
            right = i + 1
            current_len = 1  # The '/' itself
            # Expand to the left as long as the character is '1'
            while left >= 0 and S[left] == '1':
                current_len += 1
                left -= 1
            # Expand to the right as long as the character is '2'
            while right < N and S[right] == '2':
                current_len += 1
                right += 1
            # Check if the total length is odd
            if current_len % 2 == 1:
                if current_len > max_len:
                    max_len = current_len
    return max_len

# Read input
N = int(input())
S = input().strip()

# Compute the result
result = max_11_22_substring_length(N, S)

# Print the result
print(result)