def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    H = list(map(int, data[1:N+1]))
    
    # Define the lookup table for attacks needed based on base_T and rem
    lookup = [
        [0, 1, 1, 1, 2],  # base_T=0
        [0, 1, 2, 3, 3],  # base_T=1
        [0, 1, 2, 2, 2]   # base_T=2
    ]
    
    T_end = 0
    for i in range(N):
        T_start = T_end + 1
        H_i = H[i]
        cycles = H_i // 5
        rem = H_i % 5
        base_T = T_start % 3
        attacks_for_rem = lookup[base_T][rem]
        ki = cycles * 3 + attacks_for_rem
        T_end = T_start + ki - 1
    print(T_end)

if __name__ == "__main__":
    main()