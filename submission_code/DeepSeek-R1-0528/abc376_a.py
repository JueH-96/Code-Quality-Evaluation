def main():
    n, c = map(int, input().split())
    t_list = list(map(int, input().split()))
    
    if n == 0:
        print(0)
        return
        
    count = 1
    last_candy_time = t_list[0]
    
    for i in range(1, n):
        if t_list[i] - last_candy_time >= c:
            count += 1
            last_candy_time = t_list[i]
            
    print(count)

if __name__ == "__main__":
    main()