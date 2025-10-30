import itertools

n = int(input())
s = input().strip()

# Generate all permutations
perms = itertools.permutations(s)

# Convert to integers, keep unique ones, and count perfect squares
seen = set()
count = 0
for perm in perms:
    num_str = ''.join(perm)
    num = int(num_str)
    if num not in seen:
        seen.add(num)
        # Check if num is a perfect square
        sqrt = int(num ** 0.5)
        if sqrt * sqrt == num:
            count += 1

print(count)