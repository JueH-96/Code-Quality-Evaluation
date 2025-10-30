import sys

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    A = list(map(int, data[1:1+n]))
    max_val = max(A) if A else 0
    freq = [0] * (max_val + 1)
    for num in A:
        if num <= max_val:
            freq[num] += 1

    prefix = [0] * (max_val + 1)
    for i in range(1, max_val + 1):
        prefix[i] = prefix[i-1] + freq[i]

    total = 0
    for a in range(1, max_val + 1):
        if freq[a] == 0:
            continue
        k = 1
        while k * a <= max_val:
            L = k * a
            R = min((k + 1) * a - 1, max_val)
            count_in_interval = prefix[R] - prefix[L-1] if L > 0 else prefix[R]
            total += k * freq[a] * count_in_interval
            k += 1

    for a in range(1, max_val + 1):
        if freq[a] > 0:
            total -= freq[a]

    print(total)

if __name__ == "__main__":
    main()