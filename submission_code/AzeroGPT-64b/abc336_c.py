from math import log10, pow

def is_gd(num):
    for i in [int(i) for i in str(num)]:
        if i&1:return 0
    return 1

def gd_num(n):
    init=1
    for i in range(1,int(log10(n))+1):
        if 4**(i+1)>=n: init=i;break
    out=str(pow(2,init))
    diff=n-int(pow(4,init))
    while diff:
        loc=len(out)//2
        prev=int(out[loc])
        new=prev+1 if int(out,2)&1 or len(out)<init+1 else prev+2
        out=out[:loc-1]+str(new)+out[loc:]
        diff-=4**(init+1-loc)
    if int(out[-1])!=prev:
        front=0
        if len(out)<init+1: front=20
        out=str(int(out[:(len(out)+1)//2])//2)+str(int(out[(len(out)+1)//2:]))
        out=str(front)+out+str(new)+out[::-1]
    return out

N=int(input())
print(gd_num(N))