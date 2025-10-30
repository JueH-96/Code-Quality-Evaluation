n = int(input())
p = list(map(int, input().split()))

# final_pos[i] will store the final position of element (i+1)
final_pos = [0] * n

for i in range(n):
    final_pos[i] = p[i]  # Initial position where element (i+1) is inserted
    
    # Count how many elements j > i will be inserted at positions <= final_pos[i]
    for j in range(i + 1, n):
        if p[j] <= final_pos[i]:
            final_pos[i] += 1

# Construct the final array
result = [0] * n
for i in range(n):
    result[final_pos[i] - 1] = i + 1

print(' '.join(map(str, result)))