# YOUR CODE HERE
def count_abc_occurrences(s):
    count = 0
    for i in range(len(s) - 2):
        if s[i:i+3] == "ABC":
            count += 1
    return count

n, q = map(int, input().split())
s = list(input().strip())

for _ in range(q):
    x, c = input().split()
    x = int(x) - 1  # Convert to 0-based indexing
    
    # Replace character at position x with c
    s[x] = c
    
    # Count occurrences of "ABC"
    result = count_abc_occurrences(''.join(s))
    print(result)