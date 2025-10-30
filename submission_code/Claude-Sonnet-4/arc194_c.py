n = int(input())
A = list(map(int, input().split()))
B = list(map(int, input().split()))
C = list(map(int, input().split()))

# Find positions that need to be flipped
flip_to_0 = []  # positions where A[i] = 1 and B[i] = 0
flip_to_1 = []  # positions where A[i] = 0 and B[i] = 1

for i in range(n):
    if A[i] != B[i]:
        if A[i] == 1 and B[i] == 0:
            flip_to_0.append(i)
        else:  # A[i] == 0 and B[i] == 1
            flip_to_1.append(i)

# Sort flip_to_0 by decreasing cost (flip expensive ones first when they don't contribute to future costs)
flip_to_0.sort(key=lambda x: C[x], reverse=True)

# Sort flip_to_1 by increasing cost (flip cheap ones first to minimize their contribution to future costs)
flip_to_1.sort(key=lambda x: C[x])

total_cost = 0
current_A = A[:]

# First, flip all 1->0 positions
for i in flip_to_0:
    current_A[i] = 0
    # Calculate cost: sum of C[j] for all j where current_A[j] = 1
    cost = sum(C[j] for j in range(n) if current_A[j] == 1)
    total_cost += cost

# Then, flip all 0->1 positions
for i in flip_to_1:
    current_A[i] = 1
    # Calculate cost: sum of C[j] for all j where current_A[j] = 1
    cost = sum(C[j] for j in range(n) if current_A[j] == 1)
    total_cost += cost

print(total_cost)