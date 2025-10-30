import sys

def min_operations_to_remove_black_cells(t, test_cases):
    results = []
    for n, k, s in test_cases:
        i = 0
        count = 0
        while i < n:
            if s[i] == 'B':
                count += 1
                i += k
            else:
                i += 1
        results.append(count)
    return results

def main():
    input = sys.stdin.read
    data = input().split()

    index = 0
    t = int(data[index])
    index += 1
    test_cases = []

    for _ in range(t):
        n = int(data[index])
        k = int(data[index + 1])
        s = data[index + 2]
        index += 3
        test_cases.append((n, k, s))

    results = min_operations_to_remove_black_cells(t, test_cases)
    for result in results:
        sys.stdout.write(f"{result}
")

if __name__ == "__main__":
    main()