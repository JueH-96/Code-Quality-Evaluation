import sys

def main():
    N = int(sys.stdin.readline().strip())
    
    if N < 1_000:
        ans = N
    elif N < 10_000:
        ans = (N // 10) * 10
    elif N < 100_000:
        ans = (N // 100) * 100
    elif N < 1_000_000:
        ans = (N // 1_000) * 1_000
    elif N < 10_000_000:
        ans = (N // 10_000) * 10_000
    elif N < 100_000_000:
        ans = (N // 100_000) * 100_000
    else:  # 10^8 ≤ N ≤ 10^9 - 1
        ans = (N // 1_000_000) * 1_000_000
    
    print(ans)

if __name__ == "__main__":
    main()