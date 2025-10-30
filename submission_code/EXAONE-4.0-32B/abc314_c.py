import sys

def main():
    data = sys.stdin.read().splitlines()
    n, m = map(int, data[0].split())
    s = data[1].strip()
    colors = list(map(int, data[2].split()))
    
    groups = [[] for _ in range(m+1)]
    
    for i in range(n):
        c = colors[i]
        groups[c].append(i)
        
    arr = list(s)
    for c in range(1, m+1):
        indices = groups[c]
        k = len(indices)
        if k <= 1:
            continue
        last_char = arr[indices[-1]]
        for j in range(k-1, 0, -1):
            arr[indices[j]] = arr[indices[j-1]]
        arr[indices[0]] = last_char
        
    print(''.join(arr))

if __name__ == "__main__":
    main()