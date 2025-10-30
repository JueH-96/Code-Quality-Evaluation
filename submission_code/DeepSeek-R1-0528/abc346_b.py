def main():
    W, B = map(int, input().split())
    total = W + B
    base = "wbwbwwbwbwbw"
    n = len(base)
    
    for start in range(n):
        count_w = 0
        for i in range(total):
            idx = (start + i) % n
            if base[idx] == 'w':
                count_w += 1
        if count_w == W:
            print("Yes")
            return
            
    print("No")

if __name__ == "__main__":
    main()