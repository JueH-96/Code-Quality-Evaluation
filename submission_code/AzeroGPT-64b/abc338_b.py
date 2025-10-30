s=input()
from collections import Counter
c = Counter(s).most_common()
max_freq = c[0][1]
candidates = [c[0][0]]
for i in range(1, len(c)):
    if c[i][1] == max_freq:
        candidates.append(c[i][0])
    else:
        break
candidates.sort()
print(candidates[0])