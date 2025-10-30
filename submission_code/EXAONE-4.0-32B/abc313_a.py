def main():
    n = int(input().strip())
    arr = list(map(int, input().split()))
    
    if n == 1:
        print(0)
    else:
        max_others = max(arr[1:])
        if arr[0] > max_others:
            print(0)
        else:
            print(max_others - arr[0] + 1)

if __name__ == "__main__":
    main()