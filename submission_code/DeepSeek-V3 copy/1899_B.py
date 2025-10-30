def max_diff(test_cases):
    results = []
    for case in test_cases:
        n, a = case
        max_diff_val = 0
        for k in range(1, n+1):
            if n % k != 0:
                continue
            sums = []
            current_sum = 0
            for i in range(n):
                current_sum += a[i]
                if (i+1) % k == 0:
                    sums.append(current_sum)
                    current_sum = 0
            if not sums:
                continue
            min_sum = min(sums)
            max_sum = max(sums)
            current_diff = max_sum - min_sum
            if current_diff > max_diff_val:
                max_diff_val = current_diff
        results.append(max_diff_val)
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
    results = max_diff(test_cases)
    for res in results:
        print(res)

if __name__ == "__main__":
    main()