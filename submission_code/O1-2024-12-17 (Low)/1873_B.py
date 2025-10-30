def main():
    import sys
    data = sys.stdin.read().strip().split()
    t = int(data[0])
    index = 1
    for _ in range(t):
        n = int(data[index]); index += 1
        digits = list(map(int, data[index:index+n]))
        index += n
        
        max_product = 0
        # Try adding 1 to each digit in turn
        for i in range(n):
            modified_digits = digits[:]
            modified_digits[i] += 1  # add 1 to the i-th digit
            product = 1
            for d in modified_digits:
                product *= d
            if product > max_product:
                max_product = product
        
        print(max_product)

# Do not forget to call main()
main()