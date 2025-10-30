N = int(input())
A = list(map(int, input().split()))

count = 0
for i in range(1, N+1):
    # Find positions of color i
    pos = []
    for j in range(2*N):
        if A[j] == i:
            pos.append(j)
    
    # Check if there is exactly one person between them
    if pos[1] - pos[0] == 2:
        count += 1

print(count)