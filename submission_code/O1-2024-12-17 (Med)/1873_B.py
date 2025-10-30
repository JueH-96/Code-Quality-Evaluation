def main():
    import sys
    data = sys.stdin.read().strip().split()
    t = int(data[0])
    idx = 1
    for _ in range(t):
        n = int(data[idx]); idx += 1
        digits = list(map(int, data[idx:idx+n]))
        idx += n

        # Compute the maximum product after adding 1 to exactly one digit
        max_product = 0
        for i in range(n):
            product = 1
            for j in range(n):
                if j == i:
                    product *= (digits[j] + 1)
                else:
                    product *= digits[j]
            if product > max_product:
                max_product = product

        print(max_product)

# Do not forget to call main()!
main()