# Read input values
N, D = map(int, input().split())
S = input()

# Calculate the number of empty boxes initially and add the eaten cookies
empty_boxes = S.count('.')
result = empty_boxes + D

# Output the result
print(result)