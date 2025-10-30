import itertools

def main():
    n, k = map(int, input().split())
    s = input().strip()
    
    # Generate all unique permutations
    unique_perms = set(itertools.permutations(s))
    
    count = 0
    for perm in unique_perms:
        perm_str = ''.join(perm)
        valid = True
        for i in range(len(perm_str) - k + 1):
            substr = perm_str[i:i+k]
            if substr == substr[::-1]:
                valid = False
                break
        if valid:
            count += 1
    print(count)

if __name__ == "__main__":
    main()