def main():
    import sys

    data = sys.stdin.read().strip().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    MAX_A = 10**6

    # Frequency array
    freq = [0]*(MAX_A+1)
    for val in A:
        freq[val] += 1

    # Prefix sum of frequencies
    # M[i] = freq[1] + freq[2] + ... + freq[i]
    M = [0]*(MAX_A+1)
    s = 0
    for i in range(1, MAX_A+1):
        s += freq[i]
        M[i] = s

    total = 0

    # 1) Count pairs of identical values (each pair contributes ratio = 1)
    for x in range(1, MAX_A+1):
        f = freq[x]
        if f > 1:
            total += f*(f-1)//2  # number of ways to pick 2 from freq[x]

    # 2) Count pairs of distinct values (x < y), ratio = floor(y/x)
    for x in range(1, MAX_A+1):
        f = freq[x]
        if f == 0:
            continue

        limit = MAX_A // x
        r = 1
        while r <= limit:
            start = r * x
            # We only want y > x, so start at x+1 if needed
            if start <= x:
                start = x + 1
            if start > MAX_A:
                break

            end = (r + 1)*x - 1
            if end > MAX_A:
                end = MAX_A
            if start > end:
                r += 1
                continue

            # How many values lie in [start..end]?
            c = M[end] - M[start-1]
            if c > 0:
                total += f * r * c

            r += 1

    print(total)

# Do not forget to call main()
main()