import sys

def main():
    data = sys.stdin.read().split()
    idx = 0
    t = int(data[idx])
    idx += 1
    for _ in range(t):
        n = int(data[idx])
        idx += 1
        digits = list(map(int, data[idx:idx+n]))
        idx += n
        max_product = 0
        for i in range(n):
            # Add 1 to the i-th digit
            temp_digits = digits.copy()
            temp_digits[i] += 1
            # Compute product
            product = 1
            for d in temp_digits:
                product *= d
            if product > max_product:
                max_product = product
        print(max_product)

if __name__ == "__main__":
    main()