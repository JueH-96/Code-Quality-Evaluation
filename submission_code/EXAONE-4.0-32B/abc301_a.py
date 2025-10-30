def main():
    import sys
    data = sys.stdin.read().splitlines()
    n = int(data[0].strip())
    s = data[1].strip()
    
    total_t = s.count('T')
    total_a = s.count('A')
    
    if total_t > total_a:
        print('T')
    elif total_a > total_t:
        print('A')
    else:
        target = total_t
        t_count = 0
        a_count = 0
        for char in s:
            if char == 'T':
                t_count += 1
            else:
                a_count += 1
            if t_count == target:
                print('T')
                return
            if a_count == target:
                print('A')
                return

if __name__ == "__main__":
    main()