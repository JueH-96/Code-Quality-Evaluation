# Read the input from stdin
N = int(input())

# Initialize the string
s = ""

# Loop through each position in the string
for i in range(N + 1):
    # Find the smallest divisor j of N between 1 and 9 such that i is a multiple of N/j
    found = False
    for j in range(1, 10):
        if N % j == 0 and i % (N // j) == 0:
            s += str(j)
            found = True
            break
    # If no such j exists, append '-'
    if not found:
        s += '-'

# Print the answer
print(s)