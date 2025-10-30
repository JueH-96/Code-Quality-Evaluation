N = int(input())
S = input()
found = False
for i in range(N - 2):
    if S[i:i+3] == 'ABC':
        print(i + 1)
        found = True
        break
if not found:
    print(-1)