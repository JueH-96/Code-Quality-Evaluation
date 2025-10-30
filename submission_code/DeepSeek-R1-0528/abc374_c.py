def main():
    n = int(input().strip())
    arr = list(map(int, input().split()))
    total = sum(arr)
    dp = {0}
    for num in arr:
        new_dp = set()
        for s in dp:
            new_dp.add(s)
            new_dp.add(s + num)
        dp = new_dp
        
    ans = total
    for s in dp:
        candidate = max(s, total - s)
        if candidate < ans:
            ans = candidate
            
    print(ans)

if __name__ == '__main__':
    main()