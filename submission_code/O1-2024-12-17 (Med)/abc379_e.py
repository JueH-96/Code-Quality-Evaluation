def main():
    import sys

    data = sys.stdin.read().strip().split()
    N = int(data[0])
    S = data[1]

    dp = 0
    answer = 0
    
    for i in range(1, N+1):
        digit = int(S[i-1])
        dp = dp * 10 + digit * i
        answer += dp

    print(answer)

# Do not forget to call the main function
if __name__ == "__main__":
    main()