import sys

def min_weirdness(N, K, A):
    socks = [2] * N
    for a in A:
        socks[a-1] -= 1
    pairs = []
    for i in range(N):
        if socks[i] == 2:
            pairs.append(i)
        elif socks[i] == 1:
            if pairs and pairs[-1] == i-1:
                pairs.pop()
            else:
                pairs.append(i)
    total_weirdness = 0
    while len(pairs) > 1:
        total_weirdness += pairs[1] - pairs[0]
        pairs = pairs[2:]
    return total_weirdness

input = sys.stdin.read
N, K, *A = map(int, input().split())
print(min_weirdness(N, K, A))