w, b = map(int, input().split())

s = "wbwbwwbwbwbw"
n = len(s)

for i in range(n):
    for j in range(i, n):
        sub = s[i:j+1]
        w_count = sub.count('w')
        b_count = sub.count('b')
        if w_count == w and b_count == b:
            print("Yes")
            exit()

print("No")