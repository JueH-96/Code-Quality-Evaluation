n, l, r = map(int, input().split())
original = list(range(1, n+1))
part1 = original[:l-1]
part2 = original[l-1:r][::-1]
part3 = original[r:]
result = part1 + part2 + part3
print(' '.join(map(str, result)))