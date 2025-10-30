# YOUR CODE HERE
import sys
input = sys.stdin.read

def count_trailing_zeros(n):
    count = 0
    while n % 2 == 0:
        n //= 2
        count += 1
    return count

N = int(input().strip())
print(count_trailing_zeros(N))