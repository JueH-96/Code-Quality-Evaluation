def count_pairs(test_cases):
    results = []
    for a in test_cases:
        count = {}
        for num in a:
            if num in count:
                count[num] += 1
            else:
                count[num] = 1
        
        pairs = 0
        for c in count.values():
            if c > 1:
                pairs += c * (c - 1) // 2
        
        results.append(pairs)
    
    return results

def main():
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    
    t = int(data[0])
    index = 1
    test_cases = []
    
    for _ in range(t):
        n = int(data[index])
        a = list(map(int, data[index + 1].split()))
        test_cases.append(a)
        index += 2
    
    results = count_pairs(test_cases)
    
    for result in results:
        print(result)

if __name__ == "__main__":
    main()