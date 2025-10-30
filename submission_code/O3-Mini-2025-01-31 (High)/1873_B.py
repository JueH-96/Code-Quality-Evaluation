def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    t = int(data[0])
    index = 1
    results = []
    for _ in range(t):
        n = int(data[index])
        index += 1
        digits = list(map(int, data[index:index+n]))
        index += n
        max_product = 0
        # For each digit, add 1 and compute the total product
        for i in range(n):
            product = 1
            for j in range(n):
                if i == j:
                    product *= (digits[j] + 1)
                else:
                    product *= digits[j]
            max_product = max(max_product, product)
        results.append(str(max_product))
    sys.stdout.write("
".join(results))
    
if __name__ == '__main__':
    main()