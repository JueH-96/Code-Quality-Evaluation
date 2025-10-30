def main():
    import sys
    input = sys.stdin.readline

    N, D = map(int, input().split())
    S = input().strip()

    # Count the total cookies initially
    total_cookies = S.count('@')
    # After eating D cookies, D boxes become empty in addition to those already empty
    # Initially empty boxes = N - total_cookies
    # Final empty boxes = (N - total_cookies) + D
    result = (N - total_cookies) + D

    print(result)

if __name__ == "__main__":
    main()