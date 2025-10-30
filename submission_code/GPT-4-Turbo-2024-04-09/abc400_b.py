def main():
    import sys
    input = sys.stdin.read
    data = input().strip()
    N, M = map(int, data.split())
    
    LIMIT = 10**9
    sum_X = 0
    current_power = 1
    
    for i in range(M + 1):
        sum_X += current_power
        if sum_X > LIMIT:
            print("inf")
            return
        if i < M:
            current_power *= N
            if current_power > LIMIT:
                print("inf")
                return
    
    print(sum_X)

if __name__ == "__main__":
    main()