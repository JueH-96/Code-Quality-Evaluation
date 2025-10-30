import sys

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    a = list(map(int, data[1:1+n]))
    
    pos = [0] * (n+1)
    for idx, num in enumerate(a):
        pos[num] = idx
        
    swaps = []
    for i in range(n):
        if a[i] == i+1:
            continue
            
        j = pos[i+1]
        x = a[i]
        y = i+1
        
        a[i] = y
        a[j] = x
        
        pos[y] = i
        pos[x] = j
        
        swaps.append((i, j))
        
    print(len(swaps))
    for i, j in swaps:
        print(i+1, j+1)

if __name__ == "__main__":
    main()