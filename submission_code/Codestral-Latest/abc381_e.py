import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
Q = int(data[1])
S = data[2]
queries = [tuple(map(int, data[i].split())) for i in range(3, 3 + Q)]

def max_11_22_length(T):
    ones = 0
    slashes = 0
    twos = 0

    for char in T:
        if char == '1':
            ones += 1
        elif char == '/':
            slashes += 1
        elif char == '2':
            twos += 1

    return min(ones, slashes, twos) * 2 + 1

results = []
for L, R in queries:
    T = S[L-1:R]
    results.append(max_11_22_length(T))

for result in results:
    print(result)