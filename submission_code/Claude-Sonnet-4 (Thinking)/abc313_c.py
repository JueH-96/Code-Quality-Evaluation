n = int(input())
a = list(map(int, input().split()))

s = sum(a)
x = s // n
k = s % n

# Sort the array
a_sorted = sorted(a)

total_ops = 0

# First (N-k) elements should have target value x
for i in range(n - k):
    if a_sorted[i] > x:
        total_ops += a_sorted[i] - x

# Last k elements should have target value x+1  
for i in range(n - k, n):
    if a_sorted[i] > x + 1:
        total_ops += a_sorted[i] - (x + 1)

print(total_ops)