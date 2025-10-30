def main():
    n = int(input().strip())
    arr = list(map(int, input().split()))
    count = 0
    
    while True:
        positives = 0
        for x in arr:
            if x > 0:
                positives += 1
                if positives >= 2:
                    break
                    
        if positives < 2:
            break
            
        arr.sort(reverse=True)
        arr[0] -= 1
        arr[1] -= 1
        count += 1
        
    print(count)

if __name__ == "__main__":
    main()