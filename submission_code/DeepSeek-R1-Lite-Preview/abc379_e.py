def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    S = data[1]
    
    digits = [int(S[i]) for i in range(N)]
    
    total_sum = 0
    for k in range(1, N + 1):
        power = 10**(N - k + 1)
        contribution = digits[k - 1] * k * (power - 1) // 9
        total_sum += contribution
    
    print(total_sum)

if __name__ == "__main__":
    main()