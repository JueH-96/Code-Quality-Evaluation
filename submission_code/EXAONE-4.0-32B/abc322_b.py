def main():
    import sys
    data = sys.stdin.read().splitlines()
    n_m_line = data[0].split()
    N = int(n_m_line[0])
    M = int(n_m_line[1])
    S = data[1].strip()
    T = data[2].strip()
    
    is_prefix = T[:N] == S
    is_suffix = T[M-N:] == S
    
    if is_prefix and is_suffix:
        print(0)
    elif is_prefix:
        print(1)
    elif is_suffix:
        print(2)
    else:
        print(3)

if __name__ == '__main__':
    main()