# YOUR CODE HERE
n, m = map(int, input().split())
first_male = [False] * (n + 1)  # Track if family has had their first male

for i in range(m):
    a, b = input().split()
    a = int(a)
    
    if b == 'M' and not first_male[a]:
        print("Yes")
        first_male[a] = True
    else:
        print("No")