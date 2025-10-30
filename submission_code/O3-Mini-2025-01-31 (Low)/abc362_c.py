def main():
    import sys
    input = sys.stdin.readline
    
    # read number of pairs
    N = int(input().strip())
    
    intervals = []
    min_sum = 0
    max_sum = 0
    for _ in range(N):
        L, R = map(int, input().split())
        intervals.append((L, R))
        min_sum += L
        max_sum += R
    
    # If zero cannot be in range [min_sum, max_sum], then it's impossible.
    if not (min_sum <= 0 <= max_sum):
        print("No")
        return
    
    # We want to adjust from baseline L's such that sum becomes 0
    # Let X_i = L_i initially, then define slack = 0 - sum(L_i)
    slack = -min_sum  # The amount we need to distribute
    X = []
    for L, R in intervals:
        # maximum amount we can add to L is diff = R - L
        diff = R - L
        add = min(diff, slack)
        X.append(L + add)
        slack -= add
        
    # slack should be 0 if our adjustments are correct
    if slack != 0:
        # should not normally happen
        print("No")
    else:
        print("Yes")
        print(" ".join(map(str, X)))
        
if __name__ == '__main__':
    main()