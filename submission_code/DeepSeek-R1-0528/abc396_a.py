def main():
    n = int(input().strip())
    arr = list(map(int, input().split()))
    
    found = False
    for i in range(n - 2):
        if arr[i] == arr[i+1] == arr[i+2]:
            found = True
            break
            
    print("Yes" if found else "No")

if __name__ == "__main__":
    main()