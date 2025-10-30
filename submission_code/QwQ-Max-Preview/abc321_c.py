import math

def find_d_and_index(K):
    cum = 0
    for d in range(1, 11):
        if d == 1:
            count = 9
        else:
            count = math.comb(10, d)
        if cum + count >= K:
            return d, K - cum - 1
        cum += count
    return None, None

def generate_combination(d, index_in_group):
    combination = []
    prev_digit = None
    current_i = index_in_group
    for pos in range(d):
        if pos == 0:
            available = list(range(1, 10))  # 1-9, sorted increasing
        else:
            available = list(range(0, prev_digit))  # digits less than prev_digit
            available.sort()
        cumulative = 0
        for a in available:
            remaining = d - pos - 1
            m = a
            needed = remaining
            cnt = math.comb(m, needed) if m >= needed else 0
            if cumulative + cnt > current_i:
                combination.append(a)
                prev_digit = a
                current_i -= cumulative
                break
            else:
                cumulative += cnt
    return combination

K = int(input())
d, idx = find_d_and_index(K)
comb = generate_combination(d, idx)
print(''.join(map(str, comb)))