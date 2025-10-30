from collections import Counter

def is_good_string(s):
    counter = Counter(s)
    freq_counter = Counter(counter.values())

    for i in range(1, len(s) + 1):
        if freq_counter[i] != 2:
            return False

    return True

s = input()
print('Yes' if is_good_string(s) else 'No')