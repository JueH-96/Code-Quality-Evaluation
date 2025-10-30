def main():
    import sys
    input = sys.stdin.readline
    N = int(input())
    A = list(map(int, input().split()))
    ans = 0
    # Repeat until at most one positive element remains.
    while True:
        A.sort(reverse=True)
        # If the second largest is zero or negative, stop.
        if A[1] <= 0:
            break
        # Decrease the two largest by 1.
        A[0] -= 1
        A[1] -= 1
        ans += 1
    print(ans)

if __name__ == "__main__":
    main()