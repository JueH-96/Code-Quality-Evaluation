def main():
    import sys
    data = sys.stdin.read().split()
    n = int(data[0])
    A = list(map(int, data[1:1+n]))
    edges = []
    index = 1 + n
    for i in range(n-1):
        s = int(data[index])
        t = int(data[index+1])
        index += 2
        edges.append((s, t))
    
    for i in range(n-1):
        s, t = edges[i]
        count = A[i] // s
        A[i+1] += count * t
    
    print(A[-1])

if __name__ == '__main__':
    main()