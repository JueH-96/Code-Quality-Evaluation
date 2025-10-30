def main():
    import sys
    data = sys.stdin.read().strip().split()
    A, B, D = map(int, data)
    
    sequence = range(A, B + 1, D)
    print(" ".join(map(str, sequence)))
    
if __name__ == "__main__":
    main()