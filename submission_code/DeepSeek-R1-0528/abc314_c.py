import sys

def main():
    data = sys.stdin.read().splitlines()
    n, m = map(int, data[0].split())
    s = data[1].strip()
    color_arr = list(map(int, data[2].split()))
    
    groups = [[] for _ in range(m+1)]
    for i in range(n):
        c = color_arr[i]
        groups[c].append(i)
    
    arr = list(s)
    for color in range(1, m+1):
        L = groups[color]
        k = len(L)
        if k < 2:
            continue
        last_char = arr[L[-1]]
        for j in range(k-1, 0, -1):
            arr[L[j]] = arr[L[j-1]]
        arr[L[0]] = last_char
        
    print(''.join(arr))

if __name__ == "__main__":
    main()