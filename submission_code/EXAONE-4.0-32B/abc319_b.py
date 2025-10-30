def main():
    n = int(input().strip())
    res = ['-'] * (n + 1)
    
    for j in range(1, 10):
        if n % j == 0:
            step = n // j
            for i in range(0, n + 1, step):
                if res[i] == '-':
                    res[i] = str(j)
                    
    print(''.join(res))

if __name__ == "__main__":
    main()