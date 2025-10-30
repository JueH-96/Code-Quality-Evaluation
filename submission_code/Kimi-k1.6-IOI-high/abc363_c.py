import itertools

def main():
    n, k = map(int, input().split())
    s = input().strip()
    
    # Generate all unique permutations of the string
    permutations = set(itertools.permutations(s))
    unique_perms = {''.join(p) for p in permutations}
    
    count = 0
    for candidate in unique_perms:
        valid = True
        for i in range(len(candidate) - k + 1):
            substring = candidate[i:i+k]
            if substring == substring[::-1]:
                valid = False
                break
        if valid:
            count += 1
    print(count)

if __name__ == "__main__":
    main()