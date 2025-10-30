from collections import defaultdict

N = int(input())
A = list(map(int, input().split()))

answer = 0

left_count = defaultdict(int)
right_count = defaultdict(int)

# Initialize right_count with all elements from index 1 to N-1
for k in range(1, N):
    right_count[A[k]] += 1

# Process each middle position j
for j in range(1, N-1):
    # Move A[j] from right_count
    right_count[A[j]] -= 1
    
    # Add A[j-1] to left_count
    left_count[A[j-1]] += 1
    
    # Calculate contribution for position j
    for v in left_count:
        if v != A[j]:
            answer += left_count[v] * right_count[v]

print(answer)