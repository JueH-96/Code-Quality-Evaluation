def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    S = data[1] if len(data) > 1 else ""
    
    if N == 0:
        print(0)
        return
    
    sum2 = N * (N + 1) // 2
    
    ten_pow_N = 10 ** N
    
    S1 = (10 * (ten_pow_N - 1)) // 9
    sum1_part1 = (N + 1) * S1
    
    numerator = 10 * (1 - (N + 1) * ten_pow_N + N * ten_pow_N * 10)
    S2 = numerator // 81
    
    sum1 = sum1_part1 - S2
    
    total = (sum1 - sum2) // 9
    
    print(total)

if __name__ == "__main__":
    main()