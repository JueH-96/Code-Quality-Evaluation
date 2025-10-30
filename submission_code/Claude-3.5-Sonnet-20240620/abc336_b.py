# YOUR CODE HERE
def count_trailing_zeros(n):
    if n == 0:
        return 0
    count = 0
    while n % 2 == 0:
        count += 1
        n //= 2
    return count

n = int(input())
print(count_trailing_zeros(n))