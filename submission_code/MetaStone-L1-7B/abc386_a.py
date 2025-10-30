from collections import Counter

cards = list(map(int, input().split()))
count = Counter(cards)
freq = sorted(count.values())

if (freq == [1, 3]) or (freq == [2, 2]):
    print("Yes")
else:
    print("No")