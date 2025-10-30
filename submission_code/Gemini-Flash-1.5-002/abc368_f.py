# YOUR CODE HERE
def count_moves(n):
    count = 0
    i = 2
    while i * i <= n:
        if n % i == 0:
            count += 1
            while n % i == 0:
                n //= i
        i += 1
    if n > 1:
        count += 1
    return count

N = int(input())
A = list(map(int, input().split()))

nim_sum = 0
for num in A:
    nim_sum ^= count_moves(num)

if nim_sum == 0:
    print("Bruno")
else:
    print("Anna")