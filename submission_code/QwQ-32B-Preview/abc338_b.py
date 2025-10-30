S = input().strip()
counts = [0] * 26

for c in S:
    index = ord(c) - ord('a')
    counts[index] += 1

max_count = max(counts)

for index in range(26):
    if counts[index] == max_count:
        print(chr(index + ord('a')))
        break