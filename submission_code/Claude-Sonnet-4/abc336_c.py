# YOUR CODE HERE
N = int(input())

# Convert (N-1) to base 5 and map digits
def nth_good_integer(n):
    if n == 1:
        return 0
    
    # Convert (n-1) to base 5
    n -= 1
    base5_digits = []
    
    if n == 0:
        base5_digits = [0]
    else:
        while n > 0:
            base5_digits.append(n % 5)
            n //= 5
        base5_digits.reverse()
    
    # Map base 5 digits to even digits
    mapping = {0: '0', 1: '2', 2: '4', 3: '6', 4: '8'}
    result = ''.join(mapping[digit] for digit in base5_digits)
    
    return int(result)

print(nth_good_integer(N))