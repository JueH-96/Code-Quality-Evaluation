# Read the input values
n, x = map(int, input().split())
s_list = list(map(int, input().split()))

# Calculate the sum of scores <= X
total = 0
for s in s_list:
    if s <= x:
        total += s

# Output the result
print(total)