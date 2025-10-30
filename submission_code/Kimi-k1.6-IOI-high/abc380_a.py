n = input().strip()
count = [0] * 10
for c in n:
    count[int(c)] += 1
if count[1] == 1 and count[2] == 2 and count[3] == 3:
    print("Yes")
else:
    print("No")