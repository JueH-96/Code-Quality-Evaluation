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
    
    has_first_male = [False] * (n + 1)
    
    results = []
    for fam, gen in babies:
        if gen == 'F':
            results.append("No")
        else:
            if has_first_male[fam]:
                results.append("No")
            else:
                results.append("Yes")
                has_first_male[fam] = True
    
    for res in results:
        print(res)

if __name__ == "__main__":
    main()