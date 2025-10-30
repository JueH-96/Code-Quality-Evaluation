# coding: utf-8
A, B = map(int,input().split())

if A in [1,2,3] and B in [2,3,4] or A in [2,3,4] and B in [1,3,5] or A in [4,5,6] and B in [5,6,7] or A in [5,6,7] and B in [4,6,8] or A in [7,8,9] and B in [8,9,0] or A in [8,9,0] and B in [7,9]:
    print("Yes")
else:
    print("No")