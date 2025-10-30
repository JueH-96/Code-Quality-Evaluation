def check_consecutive(arr):
    min_val = min(arr)
    max_val = max(arr)
    return len(arr) == max_val - min_val + 1 and len(set(arr)) == len(arr)

N, K = map(int, input().split())
P = list(map(int, input().split()))

ans = N
for i in range(N):
    for j in range(i+K-1, N):
        indices = []
        values = []
        for k in range(i, j+1):
            indices.append(k)
            values.append(P[k])
            if len(indices) == K:
                if check_consecutive(values):
                    ans = min(ans, indices[-1] - indices[0])
                indices.pop(0)
                values.pop(0)

print(ans)