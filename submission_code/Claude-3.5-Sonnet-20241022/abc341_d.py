def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def lcm(a, b):
    return a * b // gcd(a, b)

def count_numbers_up_to(n, a, b):
    # Count numbers up to n divisible by exactly one of a and b
    return n//a + n//b - 2*(n//lcm(a,b))

def binary_search(k, n, m):
    left = 1
    right = k * max(n, m)
    
    while left < right:
        mid = (left + right) // 2
        count = count_numbers_up_to(mid, n, m)
        
        if count < k:
            left = mid + 1
        else:
            right = mid
            
    return left

N, M, K = map(int, input().split())
result = binary_search(K, N, M)
print(result)