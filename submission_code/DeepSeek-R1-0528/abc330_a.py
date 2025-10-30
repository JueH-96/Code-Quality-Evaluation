def main():
    data = input().split()
    n = int(data[0])
    l = int(data[1])
    scores = list(map(int, input().split()))
    
    count = 0
    for score in scores:
        if score >= l:
            count += 1
            
    print(count)

if __name__ == '__main__':
    main()