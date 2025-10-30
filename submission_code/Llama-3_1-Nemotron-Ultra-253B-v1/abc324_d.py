import math

n = int(input())
s = input().strip()

sorted_s = sorted(s)
max_x = int(''.join(sorted(s, reverse=True)))
x_max = math.isqrt(max_x)

result = set()

for x in range(x_max + 1):
    square = x * x
    padded = f"{square:0{n}d}"
    if sorted(padded) == sorted_s:
        result.add(square)

print(len(result))