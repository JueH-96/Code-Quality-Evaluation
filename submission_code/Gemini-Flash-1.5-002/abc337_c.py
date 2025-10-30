# YOUR CODE HERE
n = int(input())
a = list(map(int, input().split()))

ans = []
nxt = [0] * (n + 1)
for i in range(n):
    nxt[i + 1] = a[i]

st = []
for i in range(1, n + 1):
    if nxt[i] == -1:
        st.append(i)

while len(st) > 0:
    cur = st.pop(0)
    ans.append(cur)
    for i in range(1, n + 1):
        if nxt[i] == cur:
            st.append(i)

print(*ans)