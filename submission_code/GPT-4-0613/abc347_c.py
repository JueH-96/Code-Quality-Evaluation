N, A, B = map(int, input().split())
D = list(map(int, input().split()))
D.append(0)
D.sort()
week = A + B
holidays = [i % week < A for i in D]
if holidays.count(True) > A:
    print("No")
else:
    print("Yes")