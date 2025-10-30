# YOUR CODE HERE
def nth_good_integer(N):
    # Convert N to a 0-based index
    N -= 1
    
    # The number of good integers with k digits is 5^(k-1) for k > 1
    # We need to find the number of digits in the N-th good integer
    k = 1
    while N >= 5**(k-1):
        N -= 5**(k-1)
        k += 1
    
    # Now N is the 0-based index of the good integer with k digits
    # Convert N to a base-5 number and map each digit to an even digit
    good_integer = []
    for _ in range(k):
        digit = N % 5
        good_integer.append(str(digit * 2))
        N //= 5
    
    # The digits are in reverse order, so reverse them
    good_integer.reverse()
    
    # Join the list into a string and convert to an integer
    return int(''.join(good_integer))

import sys
N = int(sys.stdin.read().strip())
print(nth_good_integer(N))