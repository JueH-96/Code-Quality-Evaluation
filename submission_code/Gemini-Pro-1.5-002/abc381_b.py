s = input()
n = len(s)

if n % 2 != 0:
    print("No")
else:
    flag = True
    for i in range(n // 2):
        if s[2 * i] != s[2 * i + 1]:
            flag = False
            break
    if flag:
        counts = {}
        for char in s:
            counts[char] = counts.get(char, 0) + 1
        for count in counts.values():
            if count != 2 and count !=0:
                flag = False
                break
        
    if flag:
        
        print("Yes")
    else:
        print("No")