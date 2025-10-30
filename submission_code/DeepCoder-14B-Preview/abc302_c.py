import itertools

def main():
    n, m = map(int, input().split())
    strings = [input().strip() for _ in range(n)]
    
    def diff_count(s1, s2):
        return sum(c1 != c2 for c1, c2 in zip(s1, s2))
    
    for perm in itertools.permutations(strings):
        valid = True
        for i in range(len(perm) - 1):
            if diff_count(perm[i], perm[i+1]) != 1:
                valid = False
                break
        if valid:
            print("Yes")
            return
    print("No")

if __name__ == "__main__":
    main()