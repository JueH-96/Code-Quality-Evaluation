def main():
    import sys
    n, m = map(int, sys.stdin.readline().split())
    arr = list(map(int, sys.stdin.readline().split()))
    
    present = [False] * (n + 1)
    for num in arr:
        present[num] = True
        
    missing = []
    for i in range(1, n + 1):
        if not present[i]:
            missing.append(i)
            
    count = len(missing)
    print(count)
    print(" ".join(map(str, missing)))

if __name__ == "__main__":
    main()