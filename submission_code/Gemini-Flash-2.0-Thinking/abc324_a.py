n = int(input())
a_str = input().split()
a = [int(x) for x in a_str]

if len(set(a)) == 1:
    print("Yes")
else:
    print("No")