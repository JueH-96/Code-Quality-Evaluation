import sys

def main():
    data = sys.stdin.read().split()
    N = int(data[0])
    K = int(data[1])
    A = map(int, data[2:])
    # Sum of numbers from 1 to K
    total = K * (K + 1) // 2
    # Sum of unique A_i that are in [1, K]
    seen = set()
    s = 0
    for a in A:
        if 1 <= a <= K and a not in seen:
            seen.add(a)
            s += a
    # Answer is total minus sum of those that do appear
    print(total - s)

if __name__ == "__main__":
    main()