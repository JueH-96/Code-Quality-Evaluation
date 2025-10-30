S = input().strip()

for i, char in enumerate(S):
    if S.count(char) == 1:
        print(i + 1)
        break