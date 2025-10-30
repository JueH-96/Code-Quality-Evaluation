# Read the input values A, B, D from a single line, space-separated
A, B, D = map(int, input().split())

# Initialize an empty list to store the terms of the arithmetic sequence
terms = []

# Start with the first term of the sequence
current_term = A

# Generate terms as long as the current term does not exceed the last term B
while current_term <= B:
    # Append the current term (converted to a string) to the list
    terms.append(str(current_term))
    # Calculate the next term by adding the common difference D
    current_term += D

# Print all the collected terms, separated by spaces
# The .join() method is used to concatenate the strings in the list with a space in between
print(" ".join(terms))