def main():
    import sys
    data = sys.stdin.read().split()
    n = int(data[0])
    A = list(map(int, data[1:1+n]))
    
    pos = [0] * (n + 1)
    for idx, val in enumerate(A):
        pos[val] = idx
        
    swaps = []
    for i in range(n):
        if A[i] == i + 1:
            continue
            
        j = pos[i + 1]
        x = A[i]
        y = i + 1
        
        A[i], A[j] = A[j], A[i]
        
        pos[y] = i
        pos[x] = j
        
        swaps.append((i + 1, j + 1))
        
    print(len(swaps))
    for a, b in swaps:
        print(a, b)

if __name__ == "__main__":
    main()