def main():
    data = input().split()
    n = int(data[0])
    m = int(data[1])
    S = input().strip()
    T = input().strip()
    
    is_prefix = T[:n] == S
    is_suffix = T[m - n:] == S
    
    if is_prefix and is_suffix:
        print(0)
    elif is_prefix:
        print(1)
    elif is_suffix:
        print(2)
    else:
        print(3)

if __name__ == "__main__":
    main()