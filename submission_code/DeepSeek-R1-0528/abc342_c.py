import sys

def main():
    data = sys.stdin.read().splitlines()
    n = int(data[0])
    s = data[1].strip()
    q = int(data[2])
    mapping = [chr(ord('a') + i) for i in range(26)]
    
    index = 3
    for _ in range(q):
        parts = data[index].split()
        index += 1
        if len(parts) < 2:
            continue
        c = parts[0]
        d = parts[1]
        if c == d:
            continue
        for i in range(26):
            if mapping[i] == c:
                mapping[i] = d
                
    res = []
    for char in s:
        idx = ord(char) - ord('a')
        res.append(mapping[idx])
        
    print(''.join(res))

if __name__ == "__main__":
    main()