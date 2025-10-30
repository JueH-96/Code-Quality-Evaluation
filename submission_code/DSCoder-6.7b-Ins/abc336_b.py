def ctz(n):
    return bin(n).rstrip('0').rstrip('1').count('0')

N = int(input())
print(ctz(N))