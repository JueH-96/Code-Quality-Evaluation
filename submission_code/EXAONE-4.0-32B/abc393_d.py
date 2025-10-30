def main():
    import sys
    data = sys.stdin.read().splitlines()
    n = int(data[0].strip())
    s = data[1].strip()
    
    ones = []
    for idx, char in enumerate(s):
        if char == '1':
            ones.append(idx)
            
    k = len(ones)
    if k == 0:
        print(0)
        return
        
    b = [ones[i] - i for i in range(k)]
    b.sort()
    total_swaps = 0
    left = 0
    right = k - 1
    while left < right:
        total_swaps += b[right] - b[left]
        left += 1
        right -= 1
        
    print(total_swaps)

if __name__ == "__main__":
    main()