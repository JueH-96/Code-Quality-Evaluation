def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    A = list(map(int, input_data[1:]))

    MAX_A = 10**6

    # Frequency array
    freq = [0]*(MAX_A+1)
    for v in A:
        freq[v] += 1

    # Prefix sums of freq for range queries
    # p[i] = freq[1] + freq[2] + ... + freq[i]
    p = [0]*(MAX_A+1)
    for i in range(1, MAX_A+1):
        p[i] = p[i-1] + freq[i]

    # Compute the answer
    # 1) Contribution from pairs with the same value x (ratio = 1)
    #    sum_{x} freq[x]*(freq[x]-1)//2
    ans = 0
    for x in range(1, MAX_A+1):
        if freq[x] > 1:
            fx = freq[x]
            ans += fx*(fx-1)//2  # same-value pairs, each contributes 1

    # 2) Contribution from pairs with different values x < y: freq[x]*freq[y]*floor(y/x)
    #    We sum over x, then add freq[x] * (sum over y>x of freq[y]*floor(y/x))
    #    Use a "block" method to skip intervals where floor(y/x) is constant.
    #    For each x, we iterate over y in ascending order in chunks.
    for x in range(1, MAX_A+1):
        fx = freq[x]
        if fx == 0:
            continue
        a = x+1
        sum_val = 0
        while a <= MAX_A:
            v = a // x
            if v == 0:
                break
            # If floor(a/x) = v, then for y in [a .. b], floor(y/x) = v
            # where b = min((v+1)*x - 1, MAX_A).
            b = (v+1)*x - 1
            if b > MAX_A:
                b = MAX_A
            # Sum up freq[y] for y in [a..b]
            sum_freq = p[b] - p[a-1] if a > 1 else p[b]
            if sum_freq > 0:
                sum_val += v * sum_freq
            a = b + 1
        ans += fx * sum_val

    print(ans)