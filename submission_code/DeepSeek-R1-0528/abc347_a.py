def main():
    data = input().split()
    n = int(data[0])
    k = int(data[1])
    arr = list(map(int, input().split()))
    
    result = []
    for num in arr:
        if num % k == 0:
            result.append(num // k)
            
    print(" ".join(map(str, result)))

if __name__ == "__main__":
    main()