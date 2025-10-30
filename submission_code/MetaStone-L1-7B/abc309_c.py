import bisect

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    n = int(data[0])
    k = int(data[1])
    
    medicines = []
    index = 2
    for _ in range(n):
        a = int(data[index])
        b = int(data[index + 1])
        medicines.append((a, b))
        index += 2
    
    medicines.sort()
    sorted_a = [med[0] for med in medicines]
    suffix_sum = [0] * (n + 1)
    
    for i in range(n - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + medicines[i][1]
    
    max_a = medicines[-1][0] if n > 0 else 0
    
    low = 1
    high = max_a + 1
    
    while low < high:
        mid = (low + high) // 2
        idx = bisect.bisect_left(sorted_a, mid)
        if idx < len(sorted_a):
            current_sum = suffix_sum[idx]
        else:
            current_sum = 0
        
        if current_sum <= k:
            high = mid
        else:
            low = mid + 1
    
    print(low)

if __name__ == '__main__':
    main()