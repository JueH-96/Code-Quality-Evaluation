import collections

cards_str = input().split()
cards = [int(x) for x in cards_str]

counts = collections.Counter(cards)
frequencies = list(counts.values())
frequencies.sort(reverse=True)

if frequencies == [3, 1] or frequencies == [2, 2]:
    print("Yes")
else:
    print("No")