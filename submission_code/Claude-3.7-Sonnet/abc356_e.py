from collections import Counter

def solve(n, a):
    counter = Counter(a)
    values = sorted(counter.keys())
    
    total_sum = 0
    
    for i, val1 in enumerate(values):
        count1 = counter[val1]
        
        # Pairs with the same value (floor(val1/val1) = 1)
        if count1 > 1:
            total_sum += (count1 * (count1 - 1)) // 2
        
        for j in range(i+1, len(values)):
            val2 = values[j]
            count2 = counter[val2]
            
            # Pairs with different values (val1 < val2, so floor(val2/val1) = val2 // val1)
            total_sum += count1 * count2 * (val2 // val1)
    
    return total_sum

if __name__ == "__main__":
    n = int(input())
    a = list(map(int, input().split()))
    result = solve(n, a)
    print(result)