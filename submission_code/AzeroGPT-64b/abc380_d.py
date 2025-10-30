from bisect import bisect
S = input()
Q = int(input())
K = list(map(int, input().split()))

n = len(S)
s = S.lower() + S.upper()

cycle_len = n * 2
cycle_count = (10 ** 100) % cycle_len
pos = []
part_sizes = [n]
size = n
for _ in range(60):
    size *= 2
    part_sizes.append(size % cycle_len)
    if size >= cycle_len:
        pos.append(size - cycle_len)
        break
pregap = part_sizes[bisect(part_sizes, cycle_count) - 1]

if cycle_count > 0 and cycle_count not in pos:
    cycle_count %= cycle_count + pregap
    cycle_count += pregap
start = cycle_count + K[0] - 1

N = set(S.upper() + S.lower())
mapping = [c for c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' * 2]
for c in N:
    mapping[ord(c)] = c

answer = [mapping[start % len(s)]]
if Q > 1:
    for k in K[1:]:
        start += k - 1
        answer.append(mapping[start % len(s)])

print(''.join(answer))