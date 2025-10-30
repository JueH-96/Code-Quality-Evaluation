# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    S = data[2]
    T = data[3]
    
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