def main():
    import sys
    input = sys.stdin.readline

    # Read N and M (not really used for slicing as lengths can be derived from S)
    N, M = map(int, input().split())
    S = input().strip()
    T = input().strip()
    
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

if __name__ == "__main__":
    main()