def main():
    n = int(input().strip())
    data = list(map(int, input().split()))
    
    results = []
    for i in range(n):
        start_index = i * 7
        end_index = start_index + 7
        weekly_sum = sum(data[start_index:end_index])
        results.append(str(weekly_sum))
        
    print(" ".join(results))

if __name__ == '__main__':
    main()