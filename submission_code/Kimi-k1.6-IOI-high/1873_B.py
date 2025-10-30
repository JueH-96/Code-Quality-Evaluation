def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    t = int(input[idx])
    idx += 1
    for _ in range(t):
        n = int(input[idx])
        idx += 1
        a = list(map(int, input[idx:idx+n]))
        idx += n
        max_prod = -1
        for i in range(n):
            current = 1
            for j in range(n):
                if j == i:
                    current *= (a[j] + 1)
                else:
                    current *= a[j]
            if current > max_prod:
                max_prod = current
        print(max_prod)
        
if __name__ == "__main__":
    main()