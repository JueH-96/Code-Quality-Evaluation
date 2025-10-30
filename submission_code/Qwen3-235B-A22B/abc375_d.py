from collections import defaultdict

def main():
    import sys
    s = sys.stdin.read().strip()
    positions = defaultdict(list)
    for idx, c in enumerate(s):
        positions[c].append(idx + 1)  # 1-based positions

    total = 0
    for c in positions:
        arr = positions[c]
        m = len(arr)
        if m < 2:
            continue
        
        # Compute prefix sums
        prefix = [0] * (m + 1)
        for i in range(m):
            prefix[i + 1] = prefix[i] + arr[i]
        
        # Calculate sum_contrib
        sum_contrib = 0
        for j in range(m):
            sum_contrib += arr[j] * j - prefix[j]
        
        # Subtract number of pairs
        total += (sum_contrib - (m * (m - 1) // 2))
    
    print(total)

if __name__ == "__main__":
    main()