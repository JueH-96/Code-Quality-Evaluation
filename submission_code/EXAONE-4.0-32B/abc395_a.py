def main():
    n = int(input().strip())
    arr = list(map(int, input().split()))
    
    for i in range(n - 1):
        if arr[i] >= arr[i + 1]:
            print("No")
            return
            
    print("Yes")

if __name__ == "__main__":
    main()