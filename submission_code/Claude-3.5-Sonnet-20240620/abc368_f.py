# YOUR CODE HERE
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

def grundy(n):
    if n == 1:
        return 0
    if is_prime(n):
        return 1
    return 2

N = int(input())
A = list(map(int, input().split()))

xor_sum = 0
for num in A:
    xor_sum ^= grundy(num)

if xor_sum == 0:
    print("Bruno")
else:
    print("Anna")