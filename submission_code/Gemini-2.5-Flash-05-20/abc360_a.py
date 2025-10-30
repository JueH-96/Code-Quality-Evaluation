# YOUR CODE HERE
S = input()

index_R = S.find('R')
index_M = S.find('M')

if index_R < index_M:
    print("Yes")
else:
    print("No")