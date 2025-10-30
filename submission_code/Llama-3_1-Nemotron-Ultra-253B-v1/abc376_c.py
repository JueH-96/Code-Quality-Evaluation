n = int(input())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

sorted_T = sorted(A)
sorted_B = sorted(B)

prefix = [False] * n
prefix[0] = True
for p in range(1, n):
    if p-1 >= len(sorted_B):
        prefix[p] = False
    else:
        prefix[p] = prefix[p-1] and (sorted_B[p-1] >= sorted_T[p-1])

suffix = [False] * n
suffix[-1] = True
for p in range(n-2, -1, -1):
    if p >= len(sorted_B):
        suffix[p] = False
    else:
        suffix[p] = (sorted_B[p] >= sorted_T[p+1]) and suffix[p+1]

candidates = []
for p in range(n):
    if prefix[p] and suffix[p]:
        if p == 0:
            x_candidate = sorted_T[0]
        else:
            x_candidate = max(sorted_T[p], sorted_B[p-1])
        if p < n-1:
            if x_candidate > sorted_B[p]:
                continue
        candidates.append(x_candidate)

if not candidates:
    print(-1)
else:
    print(min(candidates))