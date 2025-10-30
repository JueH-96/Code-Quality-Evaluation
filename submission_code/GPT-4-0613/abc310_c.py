N = int(input().strip())
sticks = [input().strip() for _ in range(N)]

sticks = [''.join(sorted([s, s[::-1]])) for s in sticks]
print(len(set(sticks)))