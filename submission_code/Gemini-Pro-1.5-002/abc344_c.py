# YOUR CODE HERE
n = int(input())
a = list(map(int, input().split()))
m = int(input())
b = list(map(int, input().split()))
l = int(input())
c = list(map(int, input().split()))
q = int(input())
x = list(map(int, input().split()))

possible_sums = set()
for val_a in a:
    for val_b in b:
        for val_c in c:
            possible_sums.add(val_a + val_b + val_c)

for val_x in x:
    if val_x in possible_sums:
        print("Yes")
    else:
        print("No")