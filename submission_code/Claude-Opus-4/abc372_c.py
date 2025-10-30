# YOUR CODE HERE
def count_abc(s):
    count = 0
    for i in range(len(s) - 2):
        if s[i:i+3] == "ABC":
            count += 1
    return count

# Read input
n, q = map(int, input().split())
s = list(input().strip())

# Process queries
for _ in range(q):
    x, c = input().split()
    x = int(x) - 1  # Convert to 0-indexed
    
    # Update the string
    s[x] = c
    
    # Count occurrences of "ABC"
    count = 0
    for i in range(n - 2):
        if s[i] == 'A' and s[i+1] == 'B' and s[i+2] == 'C':
            count += 1
    
    print(count)