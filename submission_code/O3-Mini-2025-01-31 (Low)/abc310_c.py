def main():
    import sys
    input_data = sys.stdin.read().splitlines()
    n = int(input_data[0])
    unique_sticks = set()
    for i in range(1, n + 1):
        s = input_data[i].strip()
        rev = s[::-1]
        canonical = s if s < rev else rev
        unique_sticks.add(canonical)
    print(len(unique_sticks))

if __name__ == '__main__':
    main()