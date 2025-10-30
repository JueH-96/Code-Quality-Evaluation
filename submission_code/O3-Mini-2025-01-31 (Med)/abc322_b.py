def main():
    import sys
    input_data = sys.stdin.read().split()
    # The first two inputs are N and M, but we don't really need them directly.
    N = int(input_data[0])
    M = int(input_data[1])
    S = input_data[2]
    T = input_data[3]
    
    is_prefix = T.startswith(S)
    is_suffix = T.endswith(S)
    
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