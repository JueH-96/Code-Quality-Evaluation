# YOUR CODE HERE
n = int(input())
s = input()
t = input()

def are_similar(x, y):
    # Same character
    if x == y:
        return True
    # 1 and l are similar
    if (x == '1' and y == 'l') or (x == 'l' and y == '1'):
        return True
    # 0 and o are similar
    if (x == '0' and y == 'o') or (x == 'o' and y == '0'):
        return True
    return False

# Check if all corresponding characters are similar
similar = True
for i in range(n):
    if not are_similar(s[i], t[i]):
        similar = False
        break

if similar:
    print("Yes")
else:
    print("No")