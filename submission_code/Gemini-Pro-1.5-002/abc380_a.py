n = input()
counts = {}
for digit in n:
    counts[digit] = counts.get(digit, 0) + 1

if counts.get('1', 0) == 1 and counts.get('2', 0) == 2 and counts.get('3', 0) == 3 and len(n) == 6 and all(c in '123' for c in counts):
    print("Yes")
else:
    print("No")