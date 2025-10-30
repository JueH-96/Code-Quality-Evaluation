N = int(input())
S = input()

for i in range(len(S) - 1):
    if (S[i] == 'a' and S[i+1] == 'b') or (S[i] == 'b' and S[i+1] == 'a'):
        print("Yes")
        break
else:
    print("No")