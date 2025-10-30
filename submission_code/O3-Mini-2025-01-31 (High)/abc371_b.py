def main():
    import sys
    input_lines = sys.stdin.read().strip().splitlines()
    if not input_lines:
        return
    
    # Parse first line for number of families (N) and babies (M)
    first_line = input_lines[0].split()
    N = int(first_line[0])
    M = int(first_line[1])
    
    # Track first born male status for each family (indexed from 1 to N)
    first_male_given = [False] * (N + 1)  # Using index 1 to N
    
    results = []
    # Process each baby in order
    for i in range(1, M + 1):
        family_str, gender = input_lines[i].split()
        family = int(family_str)
        if gender == 'M' and not first_male_given[family]:
            results.append("Yes")
            first_male_given[family] = True
        else:
            results.append("No")
    
    print("
".join(results))

if __name__ == '__main__':
    main()