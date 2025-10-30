def main():
    import sys
    input = sys.stdin.readline
    n = int(input())
    A = list(map(int, input().split()))
    
    # Count frequencies of each integer
    from collections import Counter
    freq = Counter(A)
    
    # Find the unique element with the maximum value and record its index
    best_val = -1
    best_idx = -1
    for idx, a in enumerate(A, start=1):
        if freq[a] == 1 and a > best_val:
            best_val = a
            best_idx = idx
    
    # Output the result
    if best_idx == -1:
        print(-1)
    else:
        print(best_idx)

if __name__ == "__main__":
    main()