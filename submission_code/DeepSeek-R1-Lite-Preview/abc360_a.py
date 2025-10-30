S = input().strip()
index_R = S.index('R')
index_M = S.index('M')
if index_R < index_M:
    print("Yes")
else:
    print("No")