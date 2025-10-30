def main():
    import sys
    data = sys.stdin.read().splitlines()
    if not data:
        print(0)
        return
    n = int(data[0].strip())
    A = list(map(int, data[1].split())) if n > 0 else []
    
    dp0 = 0
    dp1 = -10**18
    
    for a in A:
        next_dp0 = max(dp0, dp1 + 2 * a)
        next_dp1 = max(dp1, dp0 + a)
        dp0, dp1 = next_dp0, next_dp1
        
    print(max(dp0, dp1))

if __name__ == "__main__":
    main()