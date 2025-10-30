# YOUR CODE HERE
def is_good_substring(s, l, r):
    # Check if substring from index l to r (0-indexed) is good
    for i in range(l, r):
        if s[i] == s[i+1]:
            return False
    return True

def flip_range(s, l, r):
    # Flip characters from index l to r (0-indexed)
    for i in range(l, r+1):
        if s[i] == '0':
            s[i] = '1'
        else:
            s[i] = '0'

# Read input
n, q = map(int, input().split())
s = list(input().strip())

# Process queries
for _ in range(q):
    query = list(map(int, input().split()))
    query_type = query[0]
    l = query[1] - 1  # Convert to 0-indexed
    r = query[2] - 1  # Convert to 0-indexed
    
    if query_type == 1:
        # Flip query
        flip_range(s, l, r)
    else:
        # Check if good string query
        if is_good_substring(s, l, r):
            print("Yes")
        else:
            print("No")