def main():
    import sys
    
    data = sys.stdin.read().strip().split()
    N, L = map(int, data[:2])
    scores = map(int, data[2:])
    
    passed = sum(1 for s in scores if s >= L)
    print(passed)

if __name__ == "__main__":
    main()