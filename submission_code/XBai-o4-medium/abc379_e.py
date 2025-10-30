def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    S = input[1]
    prev_dp = 0
    total_sum = 0
    for i in range(N):
        current_digit = int(S[i])
        current_dp = prev_dp * 10 + current_digit * (i + 1)
        total_sum += current_dp
        prev_dp = current_dp
    print(total_sum)

if __name__ == "__main__":
    main()