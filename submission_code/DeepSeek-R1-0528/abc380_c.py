import sys

def main():
    data = sys.stdin.read().splitlines()
    n, k = map(int, data[0].split())
    s = data[1].strip()
    
    blocks = []
    i = 0
    n_len = len(s)
    while i < n_len:
        if s[i] == '1':
            start = i
            j = i
            while j < n_len and s[j] == '1':
                j += 1
            end = j - 1
            blocks.append((start, end))
            i = j
        else:
            i += 1
            
    prev_start, prev_end = blocks[k-2]
    move_start, move_end = blocks[k-1]
    L = move_end - move_start + 1
    gap = move_start - prev_end - 1
    
    parts = [
        s[:prev_end+1],
        '1' * L,
        '0' * gap,
        s[move_end+1:]
    ]
    
    print(''.join(parts))

if __name__ == "__main__":
    main()