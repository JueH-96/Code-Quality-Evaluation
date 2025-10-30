def main():
    import sys
    input = sys.stdin.read().split()
    n = int(input[0])
    a = list(map(int, input[1:n+1]))
    last_occurrence = {}
    b = []
    for idx in range(n):
        current = a[idx]
        if current in last_occurrence:
            b.append(last_occurrence[current])
        else:
            b.append(-1)
        last_occurrence[current] = idx + 1  # 1-based index
    print(' '.join(map(str, b)))

if __name__ == "__main__":
    main()