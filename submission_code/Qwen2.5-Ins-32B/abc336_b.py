def ctz(N):
    count = 0
    while N & 1 == 0 and N != 0:
        count += 1
        N >>= 1
    return count

N = int(input())
print(ctz(N))