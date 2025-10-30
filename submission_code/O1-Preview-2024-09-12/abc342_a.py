# YOUR CODE HERE
S = input()
from collections import Counter
counter = Counter(S)
unique_char = [char for char, count in counter.items() if count == 1][0]
index = S.index(unique_char) + 1  # 1-based index
print(index)