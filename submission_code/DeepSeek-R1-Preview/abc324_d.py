import sys
from collections import Counter

def main():
    N = int(sys.stdin.readline().strip())
    S = sys.stdin.readline().strip()
    digits_s = Counter(S)
    count = 0

    max_x = int(10**13 ** 0.5) + 2  # Approximately 31622776 + 2

    for x in range(0, max_x + 1):
        x_squared = x * x
        if x_squared == 0:
            if all(c == '0' for c in S):
                count += 1
            continue
        s = str(x_squared)
        k = len(s)
        if k > N:
            continue
        digits_x = Counter(s)
        required_zeros = digits_x.get('0', 0) + (N - k)
        if digits_s.get('0', 0) != required_zeros:
            continue
        valid = True
        for d, cnt in digits_x.items():
            if d == '0':
                continue
            if digits_s.get(d, 0) != cnt:
                valid = False
                break
        if not valid:
            continue
        for d, cnt in digits_s.items():
            if d == '0':
                continue
            if d not in digits_x or digits_x[d] != cnt:
                valid = False
                break
        if valid:
            count += 1
    print(count)

if __name__ == '__main__':
    main()