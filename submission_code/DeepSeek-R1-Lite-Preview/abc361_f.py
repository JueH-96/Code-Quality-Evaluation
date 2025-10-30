import math

def root(N, b):
    low = 1
    high = N
    while low <= high:
        mid = (low + high) // 2
        power = mid ** b
        if power == N:
            return mid
        elif power < N:
            low = mid + 1
        else:
            high = mid - 1
    return high

def count_perfect_powers(N):
    if N < 1:
        return 0
    if N == 1:
        return 1

    b_max = 1
    while (1 << b_max) <= N:
        b_max += 1
    b_max -= 1  # floor(log2(N))

    count = [0] * (b_max + 1)  # count[2] to count[b_max]

    for b in range(2, b_max + 1):
        a_max = root(N, b)
        count[b] = a_max - 1  # exclude a=1

    # Inclusion-Exclusion: subtract higher multiples
    for b in range(b_max, 1, -1):
        multiple = 2 * b
        while multiple <= b_max:
            count[b] -= count[multiple]
            multiple += b

    total = sum(count[2:b_max+1]) + 1  # +1 for x=1
    return total

def main():
    import sys
    N_str = sys.stdin.read().strip()
    N = int(N_str)
    print(count_perfect_powers(N))

if __name__ == '__main__':
    main()