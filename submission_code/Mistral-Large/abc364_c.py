import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
X = int(data[1])
Y = int(data[2])

A = list(map(int, data[3:3+N]))
B = list(map(int, data[3+N:3+2*N]))

dishes = sorted(zip(A, B), key=lambda x: (x[0], x[1]))

total_sweetness = 0
total_saltiness = 0
count = 0

for sweetness, saltiness in dishes:
    if total_sweetness + sweetness > X or total_saltiness + saltiness > Y:
        break
    total_sweetness += sweetness
    total_saltiness += saltiness
    count += 1

print(count)