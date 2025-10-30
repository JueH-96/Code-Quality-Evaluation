# YOUR CODE HERE
n, m = map(int, input().split())
s = list(input())
c = list(map(int, input().split()))

for i in range(1, m + 1):
    indices = [j for j, x in enumerate(c) if x == i]
    if len(indices) > 1:
        temp = s[indices[-1]]
        for k in range(len(indices) - 1, 0, -1):
            s[indices[k]] = s[indices[k - 1]]
        s[indices[0]] = temp

print("".join(s))