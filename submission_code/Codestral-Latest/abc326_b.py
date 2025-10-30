# YOUR CODE HERE
def is_326_like(n):
    hundreds = n // 100
    tens = (n // 10) % 10
    ones = n % 10
    return hundreds * tens == ones

def find_smallest_326_like_number(N):
    for i in range(N, 920):
        if is_326_like(i):
            return i

# Read input
N = int(input().strip())

# Find and print the smallest 326-like number greater than or equal to N
result = find_smallest_326_like_number(N)
print(result)