from collections import Counter

S = input()
char_counts = Counter(S)
most_common = char_counts.most_common()
earliest_char = min(char for char, count in most_common if count == most_common[0][1])
print(earliest_char)