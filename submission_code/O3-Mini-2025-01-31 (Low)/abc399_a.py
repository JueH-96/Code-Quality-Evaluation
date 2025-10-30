def main():
    import sys
    input = sys.stdin.readline
    N = int(input().strip())
    S = input().strip()
    T = input().strip()
    hamming_distance = sum(1 for s, t in zip(S, T) if s != t)
    print(hamming_distance)

if __name__ == '__main__':
    main()