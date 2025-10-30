def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    S = input[ptr]
    ptr += 1
    Q = int(input[ptr])
    ptr += 1
    
    parent = [i for i in range(26)]
    
    for _ in range(Q):
        c = input[ptr]
        d = input[ptr+1]
        ptr += 2
        from_c = ord(c) - ord('a')
        to_c = ord(d) - ord('a')
        for i in range(26):
            if parent[i] == from_c:
                parent[i] = to_c
    
    result = []
    for ch in S:
        original_idx = ord(ch) - ord('a')
        mapped_idx = parent[original_idx]
        result.append(chr(mapped_idx + ord('a')))
    
    print(''.join(result))

if __name__ == "__main__":
    main()