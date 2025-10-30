n = int(input())
A = list(map(int, input().split()))
B = list(map(int, input().split()))
C = list(map(int, input().split()))

# Find positions to flip
to_flip = []
for i in range(n):
    if A[i] != B[i]:
        to_flip.append(i)

if not to_flip:
    print(0)
else:
    # Separate positions based on their original value
    flip_from_1 = []  # A[i] = 1, need to flip to 0
    flip_from_0 = []  # A[i] = 0, need to flip to 1
    
    for i in to_flip:
        if A[i] == 1:
            flip_from_1.append(i)
        else:
            flip_from_0.append(i)
    
    # Sort: flip_from_1 in decreasing order of C[i], flip_from_0 in increasing order of C[i]
    flip_from_1.sort(key=lambda i: C[i], reverse=True)
    flip_from_0.sort(key=lambda i: C[i])
    
    # Combine the order
    flip_order = flip_from_1 + flip_from_0
    
    # Simulate the flips and calculate total cost
    current_A = A[:]
    total_cost = 0
    
    for pos in flip_order:
        current_A[pos] = 1 - current_A[pos]  # Flip
        cost = sum(current_A[i] * C[i] for i in range(n))
        total_cost += cost
    
    print(total_cost)