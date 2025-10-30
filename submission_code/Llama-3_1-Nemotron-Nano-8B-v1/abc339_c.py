n = int(input())
a = list(map(int, input().split()))
prefix = [0]
current = 0
for num in a:
    current += num
    prefix.append(current)
max_neg = max(-x for x in prefix[1:]) if n > 0 else 0
s = max(0, max_neg)
print(s + prefix[-1])