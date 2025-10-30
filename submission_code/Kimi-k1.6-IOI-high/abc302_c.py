import itertools

def main():
    n, m = map(int, input().split())
    strings = [input().strip() for _ in range(n)]
    
    for perm in itertools.permutations(strings):
        valid = True
        for i in range(n-1):
            s1, s2 = perm[i], perm[i+1]
            diff = sum(c1 != c2 for c1, c2 in zip(s1, s2))
            if diff != 1:
                valid = False
                break
        if valid:
            print("Yes")
            return
    print("No")

if __name__ == "__main__":
    main()