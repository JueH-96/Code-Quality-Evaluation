# YOUR CODE HERE
def nand(a, b):
    return 1 if a == 0 or b == 0 else 0

N = int(input())
S = input()
A = [int(c) for c in S]

total_sum = 0
for i in range(N):
    current_val = A[i]
    total_sum += current_val
    for j in range(i + 1, N):
        current_val = nand(current_val, A[j])
        total_sum += current_val

print(total_sum)