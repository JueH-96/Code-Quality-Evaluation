s = input().strip()
required_indices = [1, 3, 5, 7, 9, 11, 13, 15]
if all(s[i] == '0' for i in required_indices):
    print("Yes")
else:
    print("No")