def main():
    import sys
    data = sys.stdin.read().splitlines()
    if not data:
        return
    
    n, m = map(int, data[0].split())
    babies = []
    for i in range(1, 1 + m):
        parts = data[i].split()
        family = int(parts[0])
        gender = parts[1]
        babies.append((family, gender))
    
    has_male = [False] * (n + 1)
    results = []
    
    for family, gender in babies:
        if gender == 'F':
            results.append("No")
        else:
            if not has_male[family]:
                results.append("Yes")
                has_male[family] = True
            else:
                results.append("No")
    
    for res in results:
        print(res)

if __name__ == "__main__":
    main()