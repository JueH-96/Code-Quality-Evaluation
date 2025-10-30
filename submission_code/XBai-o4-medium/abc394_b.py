n = int(input())
strings = [input() for _ in range(n)]
sorted_strings = sorted(strings, key=lambda s: len(s))
print(''.join(sorted_strings))