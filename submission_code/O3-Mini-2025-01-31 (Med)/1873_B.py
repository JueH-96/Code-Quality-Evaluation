def main():
    import sys
    data = sys.stdin.read().split()
    pos = 0
    t = int(data[pos])
    pos += 1
    results = []
    
    for _ in range(t):
        n = int(data[pos])
        pos += 1
        digits = list(map(int, data[pos:pos+n]))
        pos += n
        
        max_product = 0
        # Iterate over all positions where we add 1 exactly once.
        for i in range(n):
            temp = digits.copy()
            temp[i] += 1  # add 1 to the selected digit
            product = 1
            for d in temp:
                product *= d
            if product > max_product:
                max_product = product
                
        results.append(str(max_product))
    
    sys.stdout.write("
".join(results))

if __name__ == "__main__":
    main()