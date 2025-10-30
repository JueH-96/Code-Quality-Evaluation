def main():
    import sys
    data = sys.stdin.read().split()
    n = int(data[0])
    arr = list(map(int, data[1:1+n]))
    
    succ = [0] * (n + 1)
    front = None
    
    for i in range(n):
        pred = arr[i]
        if pred == -1:
            front = i + 1
        else:
            succ[pred] = i + 1
            
    result = []
    cur = front
    for _ in range(n):
        result.append(str(cur))
        cur = succ[cur]
        
    print(" ".join(result))

if __name__ == "__main__":
    main()