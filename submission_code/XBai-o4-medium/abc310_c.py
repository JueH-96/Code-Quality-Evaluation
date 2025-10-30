def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    unique_sticks = set()
    for i in range(1, N + 1):
        s = data[i]
        reversed_s = s[::-1]
        key = min(s, reversed_s)
        unique_sticks.add(key)
    print(len(unique_sticks))

if __name__ == "__main__":
    main()