import sys

n = int(sys.stdin.readline())
a_list = list(map(int, sys.stdin.readline().split()))

if n == 2:
    print("Yes")
else:
    a = a_list[1]
    b = a_list[0]
    is_gp = True
    for i in range(2, n):
        if a_list[i] * b != a * a_list[i-1]:
            is_gp = False
            break
    print("Yes" if is_gp else "No")