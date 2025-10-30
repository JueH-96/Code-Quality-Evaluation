n = int(input().strip())
s = input().strip()
t = input().strip()
print(sum(1 for a, b in zip(s, t) if a != b))