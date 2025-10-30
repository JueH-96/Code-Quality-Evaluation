N = int(input())

# Remove all factors of 2
while N % 2 == 0:
    N = N // 2

# Remove all factors of 3
while N % 3 == 0:
    N = N // 3

if N == 1:
    print("Yes")
else:
    print("No")