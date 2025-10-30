import sys

def main():
    data = sys.stdin.read().split()
    t = int(data[0])
    index = 1
    out_lines = []
    for _ in range(t):
        n = int(data[index])
        m = int(data[index + 1])
        index += 2
        boxes = []
        for i in range(n):
            v = int(data[index])
            p = int(data[index + 1])
            index += 2
            boxes.append((v, p))
        
        arr = []
        for (v, p) in boxes:
            diff = v - p
            if diff >= 0:
                arr.append((v, p, diff))
        
        if len(arr) < m:
            out_lines.append("0")
            continue
        
        arr.sort(key=lambda x: x[2], reverse=True)
        total = 0
        min_val = 10**18
        for i in range(m):
            total += arr[i][2]
            if arr[i][2] < min_val:
                min_val = arr[i][2]
        
        ans = total + m - min_val
        out_lines.append(str(ans))
    
    print("
".join(out_lines))

if __name__ == '__main__':
    main()