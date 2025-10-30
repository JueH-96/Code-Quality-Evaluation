# YOUR CODE HERE
def ctz(n):
    count = 0
    while n % 2 == 0 and n != 0:
        n = n // 2
        count += 1
    return count

N = int(input())
print(ctz(N))