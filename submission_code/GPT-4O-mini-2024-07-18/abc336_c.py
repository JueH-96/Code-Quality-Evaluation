def nth_good_integer(N):
    # The digits we can use to form good integers
    even_digits = ['0', '2', '4', '6', '8']
    
    # We will use a list to store the good integers
    good_integers = []
    
    # We will generate good integers in a systematic way
    # We can think of this as a base-5 number system where
    # 0 -> 0, 1 -> 2, 2 -> 4, 3 -> 6, 4 -> 8
    # We will generate numbers by treating N as a base-5 number
    while N > 0:
        N -= 1  # Adjust for 0-based indexing
        remainder = N % 5
        good_integers.append(even_digits[remainder])
        N //= 5
    
    # The good integers are generated in reverse order
    good_integers.reverse()
    
    # Join the list to form the final good integer
    return int(''.join(good_integers))

# Read input
import sys
input = sys.stdin.read
N = int(input().strip())

# Get the N-th good integer and print it
print(nth_good_integer(N))