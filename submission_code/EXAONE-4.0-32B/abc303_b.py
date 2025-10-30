def main():
    import sys
    data = sys.stdin.read().splitlines()
    if not data:
        print(0)
        return
        
    n, m = map(int, data[0].split())
    total_pairs = n * (n - 1) // 2
    seen = set()
    
    index = 1
    for _ in range(m):
        line = data[index].split()
        index += 1
        arr = list(map(int, line))
        for j in range(len(arr) - 1):
            u = arr[j]
            v = arr[j + 1]
            if u < v:
                pair = (u, v)
            else:
                pair = (v, u)
            seen.add(pair)
            
    result = total_pairs - len(seen)
    print(result)

if __name__ == "__main__":
    main()