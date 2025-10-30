from collections import Counter

A, B, C, D = map(int, input().split())
cards = [A, B, C, D]
count = Counter(cards)
if len(count) == 2:
    for num, freq in count.items():
        if freq == 3:
            print("Yes")
            exit()
elif len(count) == 3:
    for num, freq in count.items():
        if freq == 2:
            print("Yes")
            exit()
        elif freq == 3:
            print("Yes")
            exit()

print("No")