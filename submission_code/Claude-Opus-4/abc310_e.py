# YOUR CODE HERE
def nand(a, b):
    return 1 if (a, b) != (1, 1) else 0

N = int(input())
S = input().strip()

# Convert string to array of integers
A = [int(c) for c in S]

total_sum = 0

# For each starting position i (0-indexed)
for i in range(N):
    current = A[i]
    total_sum += current  # f(i,i)
    
    # For each ending position j > i
    for j in range(i + 1, N):
        current = nand(current, A[j])
        total_sum += current  # f(i,j)

print(total_sum)