# YOUR CODE HERE
s = list(map(int, input().split()))

if all(100 <= x <= 675 for x in s) and all(x % 25 == 0 for x in s) and all(s[i] <= s[i+1] for i in range(7)):
    print("Yes")
else:
    print("No")