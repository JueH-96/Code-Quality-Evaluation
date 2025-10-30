from bisect import bisect_right as br

N, M = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

Au = sorted(list(set(A)))

for b in B:
    i = br(Au, b)  # Find the rightmost index where Au[i] <= b
    if i >= len(Au):  # If b is higher than any gourmet level
        print(-1)
        continue

    # Find the person with the gourmet level Au[i]
    j = A.index(Au[i])
    A[j] = 0  # Mark this person as "has eaten sushi"
    print(j + 1)