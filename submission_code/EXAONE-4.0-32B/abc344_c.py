def main():
    import sys
    data = sys.stdin.read().split()
    idx = 0
    n = int(data[idx])
    idx += 1
    A = list(map(int, data[idx:idx+n]))
    idx += n
    
    m = int(data[idx])
    idx += 1
    B = list(map(int, data[idx:idx+m]))
    idx += m
    
    l = int(data[idx])
    idx += 1
    C = list(map(int, data[idx:idx+l]))
    idx += l
    
    q = int(data[idx])
    idx += 1
    X = list(map(int, data[idx:idx+q]))
    
    total_set = set()
    for a in A:
        for b in B:
            for c in C:
                total_set.add(a + b + c)
                
    results = []
    for x in X:
        if x in total_set:
            results.append("Yes")
        else:
            results.append("No")
            
    print("
".join(results))

if __name__ == "__main__":
    main()