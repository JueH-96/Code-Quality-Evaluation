import sys
def decimalToBinary(n):
    return bin(n).replace("0b","")
def complete(l,ourlist):
    string=""
    for num in ourlist:
        string+=str(num)
    s={int(string,2)}
    l=len(l)
    string = list(string)
    for i in range(len(l)):
        if(l[i]=='?'):
            copy=string.copy()
            copy[i]='1'
            newstring=""
            for d in copy:
                newstring+=d
            s.add(int(newstring,2))
    return s
S=input()
N=int(input())
todo=complete(S,[])
myList=sorted(todo,reverse=True)
for i in range(len(myList)):
    if(myList[i]<=N):
        print(myList[i])
        break

if(i==len(myList)-1 and myList[i]>N):
    print(-1)