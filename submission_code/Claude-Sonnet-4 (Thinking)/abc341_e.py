def is_good_range(s, l, r):
    for i in range(l, r - 1):
        if s[i] == s[i + 1]:
            return False
    return True

n, q = map(int, input().split())
s = list(input().strip())

for _ in range(q):
    query = list(map(int, input().split()))
    if query[0] == 1:  # Flip operation
        l, r = query[1], query[2]
        for i in range(l - 1, r):  # Convert to 0-indexed
            s[i] = '1' if s[i] == '0' else '0'
    else:  # Check operation
        l, r = query[1], query[2]
        if is_good_range(s, l - 1, r):  # Convert to 0-indexed
            print("Yes")
        else:
            print("No")