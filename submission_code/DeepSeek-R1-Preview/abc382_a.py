# Read the input values
n, d = map(int, input().split())
s = input().strip()

# Count the number of initially empty boxes
empty = s.count('.')

# The result is the initial empty plus D boxes being emptied
print(empty + d)