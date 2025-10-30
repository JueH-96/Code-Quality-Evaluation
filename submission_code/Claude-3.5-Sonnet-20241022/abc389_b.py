def factorial(n):
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result

# Read input
X = int(input())

# Binary search to find N where N! = X
left = 1
right = 20  # Since 21! > 3×10^18 (constraint upper bound)

while left <= right:
    mid = (left + right) // 2
    fact = factorial(mid)
    
    if fact == X:
        print(mid)
        break
    elif fact < X:
        left = mid + 1
    else:
        right = mid - 1