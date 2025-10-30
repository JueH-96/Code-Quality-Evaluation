n = int(input().strip())
A = list(map(int, input().split()))

l, r = 0, 0
seen = set()
max_len = 0

while r + 1 < n:
    if A[r] == A[r + 1] and A[r] not in seen:
        seen.add(A[r])
        r += 2
        current_length = r - l
        if current_length > max_len:
            max_len = current_length
    else:
        if l < r:
            seen.discard(A[l])
            l += 2
        else:
            l = r + 1
            r = r + 1
            seen = set()

print(max_len)