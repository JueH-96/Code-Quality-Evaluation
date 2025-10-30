S = input().strip()

# Check if all even-indexed characters (1-based) are '0'
for i in range(1, 16, 2):
    if S[i] != '0':
        print("No")
        break
else:
    print("Yes")