# YOUR CODE HERE
def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def lcm(a, b):
    return a * b // gcd(a, b)

def count_divisible(num, limit):
    return limit // num

N, M, K = map(int, input().split())

lcm_NM = lcm(N, M)
low, high = 1, 10**18

while low < high:
    mid = (low + high) // 2
    count = count_divisible(N, mid) + count_divisible(M, mid) - 2 * count_divisible(lcm_NM, mid)
    
    if count < K:
        low = mid + 1
    else:
        high = mid

result = low

while True:
    if result % N == 0 and result % M != 0:
        break
    if result % M == 0 and result % N != 0:
        break
    result += 1

print(result)