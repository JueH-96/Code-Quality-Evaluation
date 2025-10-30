def main():
    n = int(input().strip())
    arr = list(map(int, input().split()))
    
    first = arr[0]
    all_equal = True
    for num in arr:
        if num != first:
            all_equal = False
            break
            
    print("Yes" if all_equal else "No")

if __name__ == "__main__":
    main()