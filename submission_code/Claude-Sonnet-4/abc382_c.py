# Read input
N, M = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

# Process each sushi
for j in range(M):
    sushi_deliciousness = B[j]
    eaten_by = -1
    
    # Check each person in order
    for i in range(N):
        if sushi_deliciousness >= A[i]:
            eaten_by = i + 1  # Person numbers are 1-indexed
            break
    
    print(eaten_by)