def main():
    import sys
    data = sys.stdin.read().split()
    N = int(data[0])
    index = 1
    as_and_bs = []
    for _ in range(N):
        a = int(data[index])
        b = int(data[index + 1])
        as_and_bs.append((a, b))
        index += 2
    sum_A = sum(a for a, b in as_and_bs)
    max_diff = max(b - a for a, b in as_and_bs)
    result = sum_A + max_diff
    print(result)

if __name__ == "__main__":
    main()