def main():
    import sys
    data = sys.stdin.read().split()
    n = int(data[0])
    m = int(data[1])
    a = n // 2
    total_edges = a * (n - a)
    moves_available = total_edges - m
    if moves_available % 2 == 1:
        print("Aoki")
    else:
        print("Takahashi")

if __name__ == '__main__':
    main()