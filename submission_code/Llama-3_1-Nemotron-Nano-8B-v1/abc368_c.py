def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    H = list(map(int, input[1:N+1]))
    
    add = [
        [0, 3, 4],  # s=0
        [0, 1, 2],  # s=1
        [0, 1, 4]   # s=2
    ]
    
    current_T = 0
    for h in H:
        s = (current_T + 1) % 3
        min_k = float('inf')
        for r in range(3):
            a = add[s][r]
            if a >= h:
                candidate = r
            else:
                required = h - a
                q = (required + 4) // 5
                candidate = 3 * q + r
            if candidate < min_k:
                min_k = candidate
        current_T += min_k
    print(current_T)

if __name__ == "__main__":
    main()