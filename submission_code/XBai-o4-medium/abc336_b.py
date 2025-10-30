# Read input
N = int(input())

count = 0

# Keep dividing by 2 while the number is even
while N % 2 == 0:
    N = N // 2
    count += 1

# Output the count of trailing zeros
print(count)