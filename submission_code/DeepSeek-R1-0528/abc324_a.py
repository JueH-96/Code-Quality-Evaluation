def main():
    n = int(input())
    arr = list(map(int, input().split()))
    
    if len(set(arr)) == 1:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()