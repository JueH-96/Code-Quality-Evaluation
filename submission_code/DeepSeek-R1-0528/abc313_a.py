def main():
    n = int(input().strip())
    arr = list(map(int, input().split()))
    
    if n == 1:
        print(0)
    else:
        max_other = max(arr[1:])
        ans = max(0, max_other - arr[0] + 1)
        print(ans)

if __name__ == "__main__":
    main()