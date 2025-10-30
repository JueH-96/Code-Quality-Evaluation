s = input().strip()
t = input().strip()

flexible = set('atcoder@')

freq_s = {}
freq_t = {}

for c in s:
    freq_s[c] = freq_s.get(c, 0) + 1

for c in t:
    freq_t[c] = freq_t.get(c, 0) + 1

all_chars = set(freq_s.keys()) | set(freq_t.keys())

possible = True
for c in all_chars:
    if c not in flexible:
        if freq_s.get(c, 0) != freq_t.get(c, 0):
            possible = False
            break

print("Yes" if possible else "No")