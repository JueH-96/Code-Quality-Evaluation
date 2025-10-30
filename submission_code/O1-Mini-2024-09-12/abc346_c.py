import sys

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    K = int(input[1])
    A = map(int, input[2:])
    seen = set()
    sum_seen = 0
    for a in A:
        if 1 <= a <= K and a not in seen:
            seen.add(a)
            sum_seen += a
    total = K * (K + 1) // 2
    print(total - sum_seen)

if __name__ == "__main__":
    main()