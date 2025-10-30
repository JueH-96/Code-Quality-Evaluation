# YOUR CODE HERE
def nth_palindrome(N):
    if N == 1:
        return 0
    N -= 1  # Adjust because 0 is the first palindrome
    
    # Determine the length of the palindrome
    length = 1
    count = 9
    while N >= count:
        N -= count
        length += 1
        if length % 2 == 0:
            count = 9 * 10 ** (length // 2 - 1)
        else:
            count = 9 * 10 ** (length // 2)
    
    # Find the N-th palindrome of the determined length
    if length % 2 == 0:
        half_length = length // 2
        start = 10 ** (half_length - 1)
        num = start + N
        return int(str(num) + str(num)[::-1])
    else:
        half_length = length // 2
        start = 10 ** half_length
        num = start + N
        return int(str(num) + str(num)[-2::-1])

import sys
input = sys.stdin.read
N = int(input().strip())
print(nth_palindrome(N))