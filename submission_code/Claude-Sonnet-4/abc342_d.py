import math
from collections import defaultdict

def get_square_free_part(n):
    if n == 0:
        return 0
    
    result = 1
    i = 2
    while i * i <= n:
        count = 0
        while n % i == 0:
            n //= i
            count += 1
        if count % 2 == 1:
            result *= i
        i += 1
    
    if n > 1:
        result *= n
    
    return result

n = int(input())
a = list(map(int, input().split()))

# Count elements by their square-free parts
square_free_count = defaultdict(int)

for num in a:
    sf = get_square_free_part(num)
    square_free_count[sf] += 1

# Count pairs
total_pairs = 0
for count in square_free_count.values():
    if count >= 2:
        total_pairs += count * (count - 1) // 2

print(total_pairs)