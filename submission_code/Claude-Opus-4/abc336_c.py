# YOUR CODE HERE
N = int(input())

# The N-th smallest good integer (1-indexed) corresponds to (N-1) in 0-indexed
n = N - 1

# Even digits mapping: position i maps to 2*i
even_digits = [0, 2, 4, 6, 8]

# Convert n to base-5 and map to even digits
if n == 0:
    print(0)
else:
    result = []
    while n > 0:
        digit = n % 5
        result.append(str(even_digits[digit]))
        n //= 5
    
    # Reverse to get the correct order
    result.reverse()
    print(''.join(result))