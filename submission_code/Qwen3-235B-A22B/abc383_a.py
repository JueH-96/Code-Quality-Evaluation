def main():
    n = int(input())
    additions = []
    for _ in range(n):
        t, v = map(int, input().split())
        additions.append((t, v))
    
    prev_time = 0
    current_water = 0
    
    for t, v in additions:
        delta = t - prev_time
        current_water = max(current_water - delta, 0)
        current_water += v
        prev_time = t
    
    print(current_water)

if __name__ == "__main__":
    main()