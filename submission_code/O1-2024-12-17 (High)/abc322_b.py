def main():
    N, M = map(int, input().split())
    S = input().strip()
    T = input().strip()
    
    is_prefix = (T[:N] == S)
    is_suffix = (T[-N:] == S)
    
    if is_prefix and is_suffix:
        print(0)
    elif is_prefix:
        print(1)
    elif is_suffix:
        print(2)
    else:
        print(3)

main()