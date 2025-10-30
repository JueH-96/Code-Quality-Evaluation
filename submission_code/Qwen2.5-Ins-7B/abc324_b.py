# YOUR CODE HERE
import math

n = int(input())

sqrt_n = int(math.sqrt(n))
cbrt_n = int(round(n ** (1/3)))

if n == 1:
    print("Yes")
elif n % (sqrt_n ** 2) == 0 and math.log2(n / (sqrt_n ** 2)).is_integer():
    print("Yes")
elif n % (cbrt_n ** 3) == 0 and math.log3(n / (cbrt_n ** 3)).is_integer():
    print("Yes")
else:
    print("No")