import bisect

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    n = int(data[0])
    k = int(data[1])
    
    pairs = []
    index = 2
    for _ in range(n):
        a = int(data[index])
        b = int(data[index+1])
        pairs.append((a, b))
        index += 2
    
    pairs.sort()
    sorted_a = [x[0] for x in pairs]
    sorted_b = [x[1] for x in pairs]
    
    # Build suffix sum
    suffix_sum = [0] * (n + 1)
    for i in reversed(range(n)):
        suffix_sum[i] = suffix_sum[i + 1] + sorted_b[i]
    
    max_a = 0
    if n > 0:
        max_a = sorted_a[-1]
    
    left = 1
    right = max_a + 1 if n > 0 else 1
    
    while left < right:
        mid = (left + right) // 2
        idx = bisect.bisect_left(sorted_a, mid)
        total = suffix_sum[idx]
        
        if total <= k:
            right = mid
        else:
            left = mid + 1
    
    print(left)

if __name__ == "__main__":
    main()