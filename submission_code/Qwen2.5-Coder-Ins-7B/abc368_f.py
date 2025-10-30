# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
A = list(map(int, data[1:]))

# If there is any number that is not a power of 2, Anna wins
# because she can always reduce it to 2.
# If all numbers are powers of 2, Bruno wins because
# he can always reduce them to 1.

# Check if a number is a power of 2
def is_power_of_2(n):
    return (n & (n - 1)) == 0 and n != 0

# If there is any number that is not a power of 2, Anna wins
if any(not is_power_of_2(a) for a in A):
    print("Anna")
else:
    print("Bruno")