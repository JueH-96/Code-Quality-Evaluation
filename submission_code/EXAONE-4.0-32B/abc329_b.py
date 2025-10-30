def main():
    n = int(input().strip())
    arr = list(map(int, input().split()))
    
    max_val = max(arr)
    filtered = [x for x in arr if x != max_val]
    
    answer = max(filtered)
    print(answer)

if __name__ == '__main__':
    main()