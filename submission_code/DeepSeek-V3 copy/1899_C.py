def max_alternating_subarray_sum(test_cases):
    results = []
    for case in test_cases:
        n, a = case
        if n == 0:
            results.append(0)
            continue
        max_sum = a[0]
        current_sum = a[0]
        for i in range(1, n):
            if (a[i] % 2) != (a[i-1] % 2):
                current_sum += a[i]
            else:
                current_sum = a[i]
            if current_sum > max_sum:
                max_sum = current_sum
        results.append(max_sum)
    return results

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    t = int(data[idx])
    idx += 1
    test_cases = []
    for _ in range(t):
        n = int(data[idx])
        idx += 1
        a = list(map(int, data[idx:idx+n]))
        idx += n
        test_cases.append((n, a))
    results = max_alternating_subarray_sum(test_cases)
    for res in results:
        print(res)

if __name__ == "__main__":
    main()