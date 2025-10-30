from collections import Counter

def is_good_string(S):
    freq = Counter(S)
    freq_dist = {}
    for count in freq.values():
        if count in freq_dist:
            freq_dist[count] += 1
        else:
            freq_dist[count] = 1
    max_freq = max(freq.values())
    for i in range(1, max_freq + 1):
        if freq_dist.get(i, 0) not in [0, 2]:
            return "No"
    return "Yes"

S = input().strip()
print(is_good_string(S))