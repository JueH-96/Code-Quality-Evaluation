def main():
    n = int(input().strip())
    s = input().strip()
    
    total_t = s.count('T')
    total_a = s.count('A')
    
    if total_t > total_a:
        print('T')
    elif total_a > total_t:
        print('A')
    else:
        t_run = 0
        a_run = 0
        for char in s:
            if char == 'T':
                t_run += 1
            else:
                a_run += 1
            if t_run == total_t:
                print('T')
                return
            if a_run == total_a:
                print('A')
                return

if __name__ == "__main__":
    main()