import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
X = int(data[1])
Y = int(data[2])

A = list(map(int, data[3:N+3]))
B = list(map(int, data[N+3:]))

# Sort dishes by sweetness in descending order
dishes = sorted(zip(A, B), key=lambda x: x[0], reverse=True)

total_sweetness = 0
total_saltiness = 0
count = 0

for sweet, salty in dishes:
    if total_sweetness + sweet > X or total_saltiness + salty > Y:
        break
    total_sweetness += sweet
    total_saltiness += salty
    count += 1

print(count)