# YOUR CODE HERE
N, D = map(int, input().split())
S = input()

# Count initial empty boxes
empty_boxes = S.count('.')

# After D days, D more boxes will be empty
total_empty = empty_boxes + D

print(total_empty)