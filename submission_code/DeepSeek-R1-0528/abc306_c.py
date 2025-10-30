def main():
    n = int(input().strip())
    arr = list(map(int, input().split()))
    
    occ = [[] for _ in range(n+1)]
    
    for index, num in enumerate(arr):
        pos = index + 1
        occ[num].append(pos)
    
    pairs = []
    for i in range(1, n+1):
        mid_index = occ[i][1]
        pairs.append((mid_index, i))
    
    pairs.sort(key=lambda x: x[0])
    
    result = [str(num) for _, num in pairs]
    print(" ".join(result))

if __name__ == "__main__":
    main()