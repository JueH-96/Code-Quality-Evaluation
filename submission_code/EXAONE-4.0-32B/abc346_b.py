def main():
    import sys
    data = sys.stdin.readline().split()
    W = int(data[0])
    B = int(data[1])
    L_total = W + B
    
    base = "wbwbwwbwbwbw"
    T = base * 25
    n = len(T)
    
    count_w = 0
    count_b = 0
    for i in range(L_total):
        c = T[i]
        if c == 'w':
            count_w += 1
        elif c == 'b':
            count_b += 1
            
    if count_w == W and count_b == B:
        print("Yes")
        return
        
    for i in range(n - L_total):
        left_char = T[i]
        if left_char == 'w':
            count_w -= 1
        elif left_char == 'b':
            count_b -= 1
            
        right_char = T[i + L_total]
        if right_char == 'w':
            count_w += 1
        elif right_char == 'b':
            count_b += 1
            
        if count_w == W and count_b == B:
            print("Yes")
            return
            
    print("No")

if __name__ == "__main__":
    main()