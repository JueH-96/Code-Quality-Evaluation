# YOUR CODE HERE
def ctz(N):
    count = 0
    while N > 0 and N % 2 == 0:
        count += 1
        N //= 2
    return count

N = int(input().strip())
print(ctz(N))