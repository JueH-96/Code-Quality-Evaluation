# YOUR CODE HERE
def reduce_to_square_free(n):
    if n == 0:
        return 0
    
    # Remove all perfect square factors
    i = 2
    while i * i <= n:
        while n % (i * i) == 0:
            n //= (i * i)
        i += 1
    
    return n

n = int(input())
a = list(map(int, input().split()))

# Count occurrences of each square-free form
count = {}

for num in a:
    sf = reduce_to_square_free(num)
    if sf in count:
        count[sf] += 1
    else:
        count[sf] = 1

# Count pairs
result = 0
for c in count.values():
    result += c * (c - 1) // 2

print(result)