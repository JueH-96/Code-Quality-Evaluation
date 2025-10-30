def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    index = 0
    t = int(data[index])
    index += 1
    results = []
    
    for _ in range(t):
        n = int(data[index])
        index += 1
        a = list(map(int, data[index:index+n]))
        index += n
        
        max_product = 0
        for i in range(n):
            if a[i] < 9:  # Only consider increasing if the digit is less than 9
                modified_a = a[:]
                modified_a[i] += 1
                product = 1
                for num in modified_a:
                    product *= num
                max_product = max(max_product, product)
        
        results.append(max_product)
    
    for result in results:
        print(result)