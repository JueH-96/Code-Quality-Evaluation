A,B,C=map(int,input().split())
S=A+B+C
if S %2 ==1:print("No")
elif (S//2)%3==0 or S//3==max(A,B,C):print("Yes")
elif (S-A)%2==0 or (S-B)%2==0 or (S-C)%2==0:print("Yes")
else:print("No")