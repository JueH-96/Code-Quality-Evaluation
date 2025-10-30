def main():
    import sys
    input = sys.stdin.readline
    
    N = int(input())
    bets = []
    for _ in range(N):
        C = int(input())
        nums = list(map(int, input().split()))
        bets.append((C, set(nums)))
    X = int(input())
    
    # Find all persons who bet on X
    candidates = []
    for i, (C, s) in enumerate(bets, start=1):
        if X in s:
            candidates.append((i, C))
    
    if not candidates:
        # No one bet on X
        print(0)
        return
    
    # Find the minimum number of bets among those who bet on X
    min_bets = min(C for _, C in candidates)
    
    # Filter persons with that minimum
    result = [i for i, C in candidates if C == min_bets]
    result.sort()
    
    # Output
    print(len(result))
    print(*result)

if __name__ == "__main__":
    main()