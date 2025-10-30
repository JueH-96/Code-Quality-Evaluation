s = input().strip()

# Check all even positions (1-based) which are 0-based indices 1,3,5,7,9,11,13,15
for i in [1, 3, 5, 7, 9, 11, 13, 15]:
    if s[i] != '0':
        print("No")
        break
else:
    print("Yes")