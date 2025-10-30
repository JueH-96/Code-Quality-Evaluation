from collections import Counter

cards = list(map(int, input().split()))
counts = Counter(cards)

for card in set(cards):
    temp_counts = counts.copy()
    temp_counts[card] += 1
    values = list(temp_counts.values())
    if len(values) == 2 and (values[0] == 3 and values[1] == 2 or values[0] == 2 and values[1] == 3):
        print("Yes")
        exit()
print("No")