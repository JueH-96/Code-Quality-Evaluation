def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    S = input[1]
    
    prefix_sum = 0
    ans = 0
    for i in range(N):
        digit = int(S[i])
        contribution = digit * (i + 1)
        prefix_sum += contribution
        ans = ans * 10 + prefix_sum
    print(ans)

if __name__ == "__main__":
    main()