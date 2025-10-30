def main():
    import sys
    data = sys.stdin.read().split()
    n = int(data[0])
    q = int(data[1])
    treatments = list(map(int, data[2:2+q]))
    
    teeth = [False] * (n + 1)
    for i in range(1, n + 1):
        teeth[i] = True
        
    for t in treatments:
        teeth[t] = not teeth[t]
            
    result = sum(teeth[1:])
    print(result)

if __name__ == "__main__":
    main()