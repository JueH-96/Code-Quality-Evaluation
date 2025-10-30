import sys

def main():
    data = sys.stdin.read().splitlines()
    if not data:
        return
    
    n, q = map(int, data[0].split())
    s = data[1].strip()
    
    arr = []
    if n > 1:
        for i in range(n-1):
            if s[i] == s[i+1]:
                arr.append(1)
            else:
                arr.append(0)
    else:
        arr = []
    
    F = [0] * n
    for i in range(1, n):
        if i-1 < len(arr):
            F[i] = F[i-1] + arr[i-1]
        else:
            F[i] = F[i-1]
    
    output_lines = []
    for i in range(2, 2+q):
        parts = data[i].split()
        if not parts:
            continue
        l = int(parts[0])
        r = int(parts[1])
        if r - l < 1:
            output_lines.append("0")
        else:
            res = F[r-1] - F[l-1]
            output_lines.append(str(res))
    
    print("
".join(output_lines))

if __name__ == "__main__":
    main()