def main():
    import sys
    
    # Read the four integers from stdin
    A = list(map(int, sys.stdin.read().split()))
    
    # Count occurrences of each color
    from collections import Counter
    cnt = Counter(A)
    
    # For each color, we can form floor(count/2) pairs
    ans = sum(c // 2 for c in cnt.values())
    
    # Output the result
    print(ans)

if __name__ == "__main__":
    main()