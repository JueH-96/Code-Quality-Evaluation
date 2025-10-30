# YOUR CODE HERE
def get_square_free_part(n):
    if n == 0:
        return 0
    
    result = 1
    d = 2
    while d * d <= n:
        count = 0
        while n % d == 0:
            n //= d
            count += 1
        if count % 2 == 1:
            result *= d
        d += 1
    
    if n > 1:
        result *= n
    
    return result

n = int(input())
a = list(map(int, input().split()))

# Count occurrences of each square-free part
from collections import defaultdict
count = defaultdict(int)

for num in a:
    sf = get_square_free_part(num)
    count[sf] += 1

result = 0

# For each square-free part, count pairs
for sf, cnt in count.items():
    if sf == 0:
        # 0 can pair with any number (including other 0s)
        # Number of pairs with at least one 0
        zero_count = cnt
        total_count = n
        # Pairs where both are 0: zero_count * (zero_count - 1) / 2
        # Pairs where exactly one is 0: zero_count * (total_count - zero_count)
        result += zero_count * (zero_count - 1) // 2
        result += zero_count * (total_count - zero_count)
    else:
        # Non-zero square-free parts can only pair with themselves
        result += cnt * (cnt - 1) // 2

print(result)