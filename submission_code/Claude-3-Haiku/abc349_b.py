from collections import Counter

def is_good_string(s):
    counts = Counter(s)
    seen_counts = set()
    for count in counts.values():
        if count != 0 and count not in seen_counts:
            seen_counts.add(count)
            if len(seen_counts) > 2:
                return "No"
    return "Yes"

s = input()
print(is_good_string(s))