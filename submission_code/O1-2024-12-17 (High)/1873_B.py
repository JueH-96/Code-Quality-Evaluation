# YOUR CODE HERE

def main():
    import sys
    data = sys.stdin.read().strip().split()
    t = int(data[0])
    idx = 1
    for _ in range(t):
        n = int(data[idx]); idx += 1
        digits = list(map(int, data[idx:idx+n]))
        idx += n
        
        max_product = 0
        for i in range(n):
            product_value = 1
            for j in range(n):
                if j == i:
                    product_value *= (digits[j] + 1)
                else:
                    product_value *= digits[j]
            max_product = max(max_product, product_value)
        
        print(max_product)

# Do not forget to call main.
main()