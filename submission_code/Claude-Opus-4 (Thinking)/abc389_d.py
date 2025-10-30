import math

R = int(input())
R_squared = R * R
count = 0

for i in range(R):
    i_plus_half_squared = (i + 0.5) ** 2
    temp = R_squared - i_plus_half_squared
    if temp < 0:
        break
    if temp < 0.25:
        continue
    j_max = int(math.sqrt(temp) - 0.5)
    if i == 0:
        count += 1 + 2 * j_max
    else:
        count += 2 + 4 * j_max

print(count)