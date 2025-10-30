def main():
    import sys
    input = sys.stdin.readline
    
    # Read inputs: number of families and babies
    N_M = input().split()
    if len(N_M) < 2:
        N_M += input().split()
    N, M = map(int, N_M)
    
    # Create list to track if a male baby has already received Taro in each family
    # Use 1-indexed families so we create a list of N+1 booleans
    taro_assigned = [False] * (N + 1)
    
    for _ in range(M):
        parts = input().split()
        if not parts:
            parts = input().split()
        family, gender = parts
        family = int(family)
        
        if gender == "M" and not taro_assigned[family]:
            print("Yes")
            taro_assigned[family] = True
        else:
            print("No")

if __name__ == '__main__':
    main()