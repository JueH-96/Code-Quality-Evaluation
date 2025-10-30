def main():
    import sys
    data = sys.stdin.read().split()
    N = int(data[0])
    S = int(data[1])
    M = int(data[2])
    L = int(data[3])
    
    c_max = (N + 11) // 12 + 1
    b_max = (N + 11) // 8 + 1
    ans = float('inf')
    
    for c in range(c_max + 1):
        for b in range(b_max + 1):
            total_eggs = 8 * b + 12 * c
            if total_eggs >= N:
                cost = b * M + c * L
                if cost < ans:
                    ans = cost
            else:
                needed = N - total_eggs
                a = (needed + 5) // 6
                cost = a * S + b * M + c * L
                if cost < ans:
                    ans = cost
                    
    print(ans)

if __name__ == "__main__":
    main()