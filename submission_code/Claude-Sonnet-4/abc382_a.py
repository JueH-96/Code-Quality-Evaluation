# YOUR CODE HERE
N, D = map(int, input().split())
S = input().strip()

# Count initial empty boxes (boxes with '.')
initial_empty = S.count('.')

# After D days, D more boxes become empty (cookies eaten)
total_empty = initial_empty + D

print(total_empty)