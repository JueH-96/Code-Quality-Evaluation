# Read the input values
A, B, D = map(int, input().split())

terms = []
current = A
while current <= B:
    terms.append(str(current))
    current += D

# Print the terms separated by spaces
print(' '.join(terms))