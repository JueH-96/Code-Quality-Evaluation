def main():
    import sys
    data = sys.stdin.read().strip().split()
    # Parse the seven card values
    A = list(map(int, data))
    
    # Count occurrences of each card value
    from collections import Counter
    cnt = Counter(A)
    
    # Check if there exist two distinct values x, y such that
    # cnt[x] >= 3 and cnt[y] >= 2
    for x in cnt:
        if cnt[x] >= 3:
            for y in cnt:
                if y != x and cnt[y] >= 2:
                    print("Yes")
                    return
    
    # If no suitable pair (x, y) found, print No
    print("No")

if __name__ == "__main__":
    main()