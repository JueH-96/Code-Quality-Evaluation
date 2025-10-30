N = int(input().strip())
S = input().strip()

abc_set = set(['A', 'B', 'C'])

for i in range(N):
    if set(S[:i+1]) == abc_set:
        print(i+1)
        break