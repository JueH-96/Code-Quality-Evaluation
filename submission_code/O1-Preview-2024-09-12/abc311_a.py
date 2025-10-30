# YOUR CODE HERE
N = int(input())
S = input().strip()
seen = set()
for i, c in enumerate(S):
    seen.add(c)
    if len(seen) == 3:
        print(i + 1)
        break