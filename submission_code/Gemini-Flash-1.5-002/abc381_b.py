S = input()
n = len(S)
if n % 2 != 0:
    print("No")
else:
    flag1 = True
    for i in range(0, n, 2):
        if S[i] != S[i+1]:
            flag1 = False
            break
    
    count = {}
    for char in S:
        count[char] = count.get(char, 0) + 1
    
    flag2 = True
    for char in count:
        if count[char] != 2:
            flag2 = False
            break
    
    if flag1 and flag2:
        print("Yes")
    else:
        print("No")