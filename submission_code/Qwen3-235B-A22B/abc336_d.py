def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    n = int(data[0])
    a = list(map(int, data[1:]))

    # Precompute prefix_min and suffix_min
    prefix = [0] * n
    prefix[0] = a[0] - 0
    for i in range(1, n):
        prefix[i] = min(prefix[i-1], a[i] - i)
    
    suffix = [0] * n
    suffix[-1] = a[-1] + (n-1)
    for i in range(n-2, -1, -1):
        suffix[i] = min(suffix[i+1], a[i] + i)
    
    max_k = 0
    for c in range(n):
        left = prefix[c] + c
        right = suffix[c] - c
        total = min(left, right)
        allowed = min(c + 1, n - c)
        candidate = min(total, allowed)
        if candidate > max_k:
            max_k = candidate
    
    print(max_k)

if __name__ == "__main__":
    main()