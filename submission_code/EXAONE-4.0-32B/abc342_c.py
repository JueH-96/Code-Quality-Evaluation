import sys

def main():
    data = sys.stdin.read().splitlines()
    n = int(data[0])
    s = data[1].strip()
    q = int(data[2])
    
    mapping = list(range(26))
    
    for i in range(3, 3 + q):
        parts = data[i].split()
        if len(parts) < 2:
            continue
        c = parts[0]
        d = parts[1]
        c_idx = ord(c) - ord('a')
        d_idx = ord(d) - ord('a')
        for j in range(26):
            if mapping[j] == c_idx:
                mapping[j] = d_idx
                
    res = []
    for char in s:
        orig_index = ord(char) - ord('a')
        new_char = chr(mapping[orig_index] + ord('a'))
        res.append(new_char)
        
    print(''.join(res))

if __name__ == '__main__':
    main()