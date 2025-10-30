def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    t = int(input_data[0])
    index = 1
    results = []
    
    for _ in range(t):
        n = int(input_data[index])
        index += 1
        digits = list(map(int, input_data[index:index+n]))
        index += n
        
        max_product = 0
        # Try adding 1 to each digit exactly once.
        for i in range(n):
            modified = digits.copy()
            modified[i] += 1
            prod = 1
            for d in modified:
                prod *= d
            if prod > max_product:
                max_product = prod
        results.append(str(max_product))
        
    sys.stdout.write("
".join(results))

if __name__ == "__main__":
    main()