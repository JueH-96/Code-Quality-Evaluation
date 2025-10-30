def max_product(test_cases):
    results = []
    for case in test_cases:
        n, digits = case
        max_prod = 0
        for i in range(n):
            temp = digits.copy()
            temp[i] += 1
            product = 1
            for num in temp:
                product *= num
            if product > max_prod:
                max_prod = product
        results.append(max_prod)
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
        digits = list(map(int, data[idx:idx+n]))
        idx += n
        test_cases.append((n, digits))
    results = max_product(test_cases)
    for res in results:
        print(res)

if __name__ == "__main__":
    main()