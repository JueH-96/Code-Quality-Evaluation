t = int(input())
for _ in range(t):
    n = int(input())
    if n % 3 == 0:
        print("Second")
    elif (1000 - n) % 6 == 0:
        print("Second")
    else:
        print("First")