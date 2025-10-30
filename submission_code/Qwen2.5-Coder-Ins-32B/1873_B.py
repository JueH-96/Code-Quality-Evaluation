# YOUR CODE HERE
import sys

def max_product_with_one_increment(test_cases):
    results = []
    for case in test_cases:
        n = case[0]
        digits = case[1]
        max_product = 0
        for i in range(n):
            new_digits = digits[:]
            new_digits[i] += 1
            product = 1
            for digit in new_digits:
                product *= digit
            max_product = max(max_product, product)
        results.append(max_product)
    return results

def main():
    import sys
    input = sys.stdin.read().split()
    index = 0
    t = int(input[index])
    index += 1
    test_cases = []
    for _ in range(t):
        n = int(input[index])
        index += 1
        digits = list(map(int, input[index:index + n]))
        index += n
        test_cases.append((n, digits))
    results = max_product_with_one_increment(test_cases)
    for result in results:
        print(result)

if __name__ == "__main__":
    main()