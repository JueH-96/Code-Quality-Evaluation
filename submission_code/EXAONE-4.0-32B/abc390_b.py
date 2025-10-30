def main():
    n = int(input().strip())
    arr = list(map(int, input().split()))
    
    if n == 2:
        print("Yes")
    else:
        valid = True
        for i in range(n - 2):
            if arr[i+1] * arr[i+1] != arr[i] * arr[i+2]:
                valid = False
                break
        print("Yes" if valid else "No")

if __name__ == "__main__":
    main()