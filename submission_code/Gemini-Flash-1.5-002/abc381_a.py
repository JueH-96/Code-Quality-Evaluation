n = int(input())
s = input()

if n % 2 == 0:
    print("No")
else:
    mid = (n + 1) // 2
    ones = s[:mid - 1]
    slash = s[mid - 1]
    twos = s[mid:]
    if slash != "/" or len(ones) + len(twos) + 1 != n:
        print("No")
    elif all(c == '1' for c in ones) and all(c == '2' for c in twos):
        print("Yes")
    else:
        print("No")