# YOUR CODE HERE
s = input()
for i in range(2, 17, 2):
    if s[i-1] != '0':
        print('No')
        break
else:
    print('Yes')