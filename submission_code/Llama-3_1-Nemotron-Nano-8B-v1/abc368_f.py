max_A = 10**5
divisors = [[] for _ in range(max_A + 1)]

for d in range(2, max_A + 1):
    for multiple in range(2 * d, max_A + 1, d):
        divisors[multiple].append(d)

G = [0] * (max_A + 1)
G[1] = 0

for a in range(2, max_A + 1):
    proper = [1]
    proper.extend(divisors[a])
    s = set()
    for x in proper:
        s.add(G[x])
    mex = 0
    while mex in s:
        mex += 1
    G[a] = mex

n = int(input())
A = list(map(int, input().split()))
xor_sum = 0
for num in A:
    xor_sum ^= G[num]
print("Anna" if xor_sum != 0 else "Bruno")