N = int(input())
S = input()

def nand(a, b):
    return 1 if (a == 0 or b == 0) else 0

# Convert string to list of integers
A = [int(c) for c in S]

total = 0
for i in range(N):
    # For each starting position i
    curr = A[i]  # Current value starts as A[i]
    total += curr  # Add f(i,i)
    
    # Compute f(i,j) for all j > i
    for j in range(i+1, N):
        curr = nand(curr, A[j])  # Update running NAND value
        total += curr  # Add f(i,j)

print(total)