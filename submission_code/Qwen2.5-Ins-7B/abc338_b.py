# YOUR CODE HERE
from collections import Counter

S = input()
counter = Counter(S)
most_common = counter.most_common()

max_count = most_common[0][1]
result = most_common[0][0]

for char, count in most_common:
    if count < max_count:
        break
    if char < result:
        result = char

print(result)