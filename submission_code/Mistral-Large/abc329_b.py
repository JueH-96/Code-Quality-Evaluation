import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
A = list(map(int, data[1:]))

# Find the largest and second largest elements
largest = max(A)
second_largest = float('-inf')

for num in A:
    if num != largest:
        second_largest = max(second_largest, num)

print(second_largest)