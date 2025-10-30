s = input().strip()
Q = int(input())
Ks = list(map(int, input().split()))
original_len = len(s)

def process(k):
    current_len = original_len
    while current_len < k:
        current_len *= 2
    count_flips = 0
    while current_len > original_len:
        half = current_len // 2
        if k > half:
            k -= half
            count_flips += 1
        current_len = half
    c = s[k-1]
    if count_flips % 2 == 1:
        if c.isupper():
            return c.lower()
        else:
            return c.upper()
    else:
        return c

results = [process(k) for k in Ks]
print(' '.join(results))