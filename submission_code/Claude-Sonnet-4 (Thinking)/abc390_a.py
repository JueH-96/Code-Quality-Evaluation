A = list(map(int, input().split()))

target = [1, 2, 3, 4, 5]

if A == target:
    print("No")
else:
    found = False
    
    for i in range(4):
        temp = A.copy()
        temp[i], temp[i+1] = temp[i+1], temp[i]
        
        if temp == target:
            found = True
            break
    
    print("Yes" if found else "No")