# YOUR CODE HERE
def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def lcm(a, b):
    return (a * b) // gcd(a, b)

def count(n, m, k):
    l = lcm(n, m)
    cnt = (k // n) + (k // m) - 2 * (k // l)
    return cnt

n, m, k = map(int, input().split())

left = 1
right = 10**18

while left < right:
    mid = (left + right) // 2
    if count(n, m, mid) < k:
        left = mid + 1
    else:
        right = mid

ans = left

while True:
    if (ans % n == 0 and ans % m != 0) or (ans % n != 0 and ans % m == 0):
        k -= 1

    if k == 0:
        print(ans)
        break
    
    ans +=1