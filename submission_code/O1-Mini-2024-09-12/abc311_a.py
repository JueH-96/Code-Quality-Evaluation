N = int(input())
S = input()

seen = set()
for idx, char in enumerate(S):
    seen.add(char)
    if len(seen) == 3:
        print(idx + 1)
        break