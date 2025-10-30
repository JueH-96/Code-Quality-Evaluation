n = int(input())
s = input()
t = input()
print(sum(1 for a, b in zip(s, t) if a != b))