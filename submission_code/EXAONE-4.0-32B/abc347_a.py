def main():
    import sys
    data = sys.stdin.read().split()
    n = int(data[0])
    k = int(data[1])
    a = list(map(int, data[2:2+n]))
    
    res = []
    for num in a:
        if num % k == 0:
            res.append(str(num // k))
            
    print(" ".join(res))

if __name__ == "__main__":
    main()