# Reading input
L, R = map(int, input().split())

# Decision based on the values of L and R
if L == 1 and R == 0:
    print("Yes")
elif L == 0 and R == 1:
    print("No")
else:
    print("Invalid")