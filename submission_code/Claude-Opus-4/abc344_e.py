# YOUR CODE HERE
n = int(input())
a = list(map(int, input().split()))

# Create a dictionary to store value -> index mapping
pos = {}
for i in range(len(a)):
    pos[a[i]] = i

q = int(input())

for _ in range(q):
    query = list(map(int, input().split()))
    
    if query[0] == 1:  # Insert y after x
        x, y = query[1], query[2]
        # Find position of x
        idx = pos[x]
        # Insert y after x
        a.insert(idx + 1, y)
        # Update positions for all elements after the insertion point
        for i in range(idx + 1, len(a)):
            pos[a[i]] = i
    
    else:  # Remove x
        x = query[1]
        # Find position of x
        idx = pos[x]
        # Remove x
        a.pop(idx)
        # Remove from position dictionary
        del pos[x]
        # Update positions for all elements after the removal point
        for i in range(idx, len(a)):
            pos[a[i]] = i

print(' '.join(map(str, a)))