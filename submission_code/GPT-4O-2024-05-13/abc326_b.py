# YOUR CODE HERE
def find_326_like_number(N):
    for num in range(N, 920):
        hundreds = num // 100
        tens = (num // 10) % 10
        ones = num % 10
        if hundreds * tens == ones:
            return num

import sys
input = sys.stdin.read
N = int(input().strip())
print(find_326_like_number(N))