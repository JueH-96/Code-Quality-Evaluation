def main():
    n = int(input().strip())
    s = input().strip()
    
    dp0 = 0
    dp1 = 0
    total = 0
    
    for c in s:
        if c == '0':
            new_dp0 = 1
            new_dp1 = dp0 + dp1
        else:
            new_dp0 = dp1
            new_dp1 = dp0 + 1
        
        dp0, dp1 = new_dp0, new_dp1
        total += dp1
        
    print(total)

if __name__ == '__main__':
    main()