# Read the input
n = int(input())
s = input()
t = input()

# Check if the strings are similar
def are_similar(x, y):
    if x == y:
        return True
    if (x == '1' and y == 'l') or (x == 'l' and y == '1'):
        return True
    if (x == '0' and y == 'o') or (x == 'o' and y == '0'):
        return True
    return False

similar = True
for i in range(n):
    if not are_similar(s[i], t[i]):
        similar = False
        break

# Print the result
print("Yes" if similar else "No")