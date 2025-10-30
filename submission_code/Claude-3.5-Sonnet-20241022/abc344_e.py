N = int(input())
A = list(map(int, input().split()))
Q = int(input())

# Convert list to doubly linked list using dict
# Each element maps to [prev, next]
links = {}
for i in range(N):
    prev_idx = i-1 if i > 0 else None
    next_idx = i+1 if i < N-1 else None
    links[A[i]] = [prev_idx, next_idx]

# Keep track of head
head = A[0]

for _ in range(Q):
    query = list(map(int, input().split()))
    
    if query[0] == 1:
        x, y = query[1], query[2]
        # Get next element after x
        next_elem = None if links[x][1] is None else A[links[x][1]]
        
        # Update links for x
        links[x][1] = len(A)
        
        # Add y to A and links
        A.append(y)
        links[y] = [links[x][1], None if next_elem is None else A.index(next_elem)]
        
        # Update next element's prev pointer if exists
        if next_elem is not None:
            links[next_elem][0] = len(A)-1
            
    else:
        x = query[1]
        # Get prev and next elements
        prev_idx = links[x][0]
        next_idx = links[x][1]
        prev_elem = None if prev_idx is None else A[prev_idx]
        next_elem = None if next_idx is None else A[next_idx]
        
        # Update head if needed
        if x == head:
            head = next_elem
            
        # Update links
        if prev_elem is not None:
            links[prev_elem][1] = next_idx
        if next_elem is not None:
            links[next_elem][0] = prev_idx
            
        # Remove x from links
        del links[x]

# Build result by following next pointers from head
result = []
curr = head
while curr is not None:
    result.append(curr)
    next_idx = links[curr][1]
    curr = None if next_idx is None else A[next_idx]

print(*result)