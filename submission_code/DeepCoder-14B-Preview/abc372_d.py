# Read the input
n, q = map(int, input().split())
s = list(input().strip())

# Calculate the initial count of "ABC" substrings
current_count = 0
for k in range(len(s) - 2):
    if s[k] == 'A' and s[k+1] == 'B' and s[k+2] == 'C':
        current_count += 1

# Process each query
for _ in range(q):
    x, c = input().split()
    x = int(x) - 1  # Convert to 0-based index

    # Determine the range of triplets affected by this change
    start_k = max(0, x - 2)
    end_k = min(x, len(s) - 3)

    # Subtract existing "ABC" substrings in the affected range
    for k in range(start_k, end_k + 1):
        if s[k] == 'A' and s[k+1] == 'B' and s[k+2] == 'C':
            current_count -= 1

    # Update the character
    s[x] = c

    # Add new "ABC" substrings in the affected range
    for k in range(start_k, end_k + 1):
        if s[k] == 'A' and s[k+1] == 'B' and s[k+2] == 'C':
            current_count += 1

    # Output the current count
    print(current_count)