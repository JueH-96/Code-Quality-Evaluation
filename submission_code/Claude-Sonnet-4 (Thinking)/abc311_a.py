n = int(input())
s = input()

seen = set()
for i in range(n):
    seen.add(s[i])
    if len(seen) == 3:  # All A, B, C have been seen
        print(i + 1)  # 1-indexed position
        break