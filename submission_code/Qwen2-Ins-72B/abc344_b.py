lst = []
x = input()
while x != "0":
    lst.append(x)
    x = input()
lst.append("0")
for i in range(len(lst)-1,-1,-1):
    print(lst[i])