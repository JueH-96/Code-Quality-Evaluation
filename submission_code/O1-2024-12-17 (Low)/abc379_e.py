def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    S = data[1]
    
    dp = 0  # will hold sum of substrings ending at current position
    ans = 0  # final answer
    
    for i, ch in enumerate(S):
        digit = int(ch)
        dp = dp * 10 + (i+1) * digit
        ans += dp
    
    print(ans)

# Don't forget to call main()
if __name__ == "__main__":
    main()