# YOUR CODE HERE
def nand(a, b):
    return 1 - (a & b)

n = int(input())
s = input().strip()

# Convert string to array of integers
a = [int(c) for c in s]

total = 0

# For each starting position i
for i in range(n):
    current = a[i]  # f(i,i) = a[i]
    total += current
    
    # For each ending position j > i
    for j in range(i + 1, n):
        current = nand(current, a[j])  # f(i,j) = f(i,j-1) NAND a[j]
        total += current

print(total)