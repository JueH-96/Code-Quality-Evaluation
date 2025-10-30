def main():
    import sys
    bits = list(map(int, sys.stdin.read().split()))
    answer = sum(bits[i] * (1 << i) for i in range(64))
    print(answer)

if __name__ == "__main__":
    main()