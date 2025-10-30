import math
n=int(input())
if n%2==1:
    print("No")
else:
    s=n
    p=0
    while s%2==0:
        s=s/2
        p+=1
    if s%3==1:
        print("Yes")
    else:
        q=0
        while s%3==0:
            s=s/3
            q+=1
        if s==1:
            print("Yes")
        elif math.pow(2,p)*math.pow(3,q)==n:
            print("Yes")
        else:
            print("No")