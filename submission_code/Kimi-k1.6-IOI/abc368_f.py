max_num = 10**5

# Precompute proper divisors for each number up to max_num
divisors = [[] for _ in range(max_num + 1)]
for d in range(1, max_num + 1):
    for multiple in range(2 * d, max_num + 1, d):
        divisors[multiple].append(d)

# Precompute Grundy numbers for each number up to max_num
Grundy = [0] * (max_num + 1)
for x in range(2, max_num + 1):
    s = set()
    for d in divisors[x]:
        s.add(Grundy[d])
    mex = 0
    while mex in s:
        mex += 1
    Grundy[x] = mex

# Read input and compute XOR sum of Grundy numbers
n = int(input())
a = list(map(int, input().split()))
xor_sum = 0
for num in a:
    xor_sum ^= Grundy[num]

print("Anna" if xor_sum != 0 else "Bruno")