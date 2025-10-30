B = int(input())
for A in range(1, 101):
    if A**A == B:
        print(A)
        exit()
print(-1)