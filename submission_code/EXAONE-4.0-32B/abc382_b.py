def main():
    import sys
    data = sys.stdin.read().splitlines()
    if not data:
        return
    
    n, d = map(int, data[0].split())
    s = data[1].strip()
    
    positions = []
    for idx, char in enumerate(s):
        if char == '@':
            positions.append(idx)
            
    total_cookies = len(positions)
    remaining_count = total_cookies - d
    remaining_set = set(positions[:remaining_count])
    
    res_list = []
    for i in range(n):
        if i in remaining_set:
            res_list.append('@')
        else:
            res_list.append('.')
            
    print(''.join(res_list))

if __name__ == "__main__":
    main()