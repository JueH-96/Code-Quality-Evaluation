def main():
    import sys
    data = sys.stdin.read().splitlines()
    n = int(data[0])
    A = list(map(int, data[1].split()))
    B = list(map(int, data[2].split()))
    
    max_a = max(A)
    max_b = max(B)
    
    print(max_a + max_b)

if __name__ == "__main__":
    main()