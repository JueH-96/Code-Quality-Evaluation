def main():
    import sys
    data = sys.stdin.read().split()
    n = int(data[0])
    P = list(map(int, data[1:1+n]))
    q = int(data[1+n])
    
    pos = [0] * (n + 1)
    for index in range(n):
        person = P[index]
        pos[person] = index
        
    queries = []
    index_ptr = 1 + n + 1
    for i in range(q):
        a = int(data[index_ptr])
        b = int(data[index_ptr + 1])
        index_ptr += 2
        queries.append((a, b))
        
    results = []
    for a, b in queries:
        if pos[a] < pos[b]:
            results.append(str(a))
        else:
            results.append(str(b))
            
    print("
".join(results))

if __name__ == "__main__":
    main()